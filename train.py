import os
import time
import torch
import utils as utils
import torch.nn as nn
from transformers import get_linear_schedule_with_warmup
from transformers import AdamW
warmup_updates = 4000
from tqdm import tqdm
import json
from torch.nn import functional as F
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda")

def init_weights(m):
    if type(m) == nn.Linear:
        with torch.no_grad():
            torch.nn.init.kaiming_normal_(m.weight)

def instance_bce(logits, labels):
    assert logits.dim() == 2
    cross_entropy_loss = nn.CrossEntropyLoss()

    prediction_ans_k, top_ans_ind = torch.topk(F.softmax(labels, dim=-1), k=1, dim=-1, sorted=False)
    ce_loss = cross_entropy_loss(logits, top_ans_ind.squeeze(-1))

    return ce_loss

# multi-label soft loss
def instance_bce_with_logits(logits, labels, reduction='mean'):
    assert logits.dim() == 2
    loss = nn.functional.binary_cross_entropy_with_logits(logits, labels, reduction=reduction)
    if reduction == 'mean':
        loss *= labels.size(1)
    return loss      
            
            

def cal_acc_multi(ground_truth, preds, return_id = False):
    all_num = len(ground_truth)
    acc_num = 0
    ids = []
    temp = []
    for i, answer_id in enumerate(ground_truth):
        pred = preds[i]
        # ids.append([i, int(pred)])
        cnt = 0
        for aid in answer_id:
            if pred == aid:
                cnt += 1
        if cnt ==1:
            acc_num += 0.3
            # ids.append([int(pred), 1])
        elif cnt == 2:
            acc_num += 0.6   
            # ids.append([int(pred), 1])
        elif cnt > 2:
            acc_num += 1
            # ids.append([int(pred), 1])
        # else:
        #     ids.append([int(pred), 0])
    if return_id:
        return acc_num / all_num, ids
    else:
        return acc_num / all_num            
            
            
            
            
def compute_score_with_logits(logits, labels):
    logits = torch.max(logits, 1)[1].data # argmax
    one_hots = torch.zeros(*labels.size()).to(logits.device)
    one_hots.scatter_(1, logits.view(-1, 1), 1)
    scores = (one_hots * labels)
    return scores


def train(opt, model, train_loader, eval_loader, num_epochs, output, s_epoch=0):
    saving_epoch = 0
    grad_clip = opt.clip_norm
    bert_optim = None
    utils.create_dir(output)


    batch_per_epoch = int(len(train_loader.dataset) / opt.batch_size) + 1
    t_total = int(batch_per_epoch * opt.epochs)
    optimizer = torch.optim.Adamax(model.parameters(), lr=opt.lxmert_lr)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0.01 * t_total, num_training_steps=t_total)

    N = len(train_loader.dataset)
    num_batches = int(N / opt.batch_size + 1)
    print("num_batches",num_batches)
    logger = utils.Logger(os.path.join(output, 'log.txt'))
    logger.write(opt.__repr__())
    best_eval_score = 0.60

    utils.print_model(model, logger)

    # update_freq = int(opt.update_freq)
    wall_time_start = time.time()
    for epoch in range(s_epoch, num_epochs):
        total_loss = 0
        train_score = 0
        total_norm = 0
        count_norm = 0
        num_updates = 0
        Loss_1 = 0 
        Loss_3 = 0 
        Ans_loss = 0
        Atten_loss = 0
        Ban_loss = 0
        t = time.time()


        for i, (v, b, q, a, qid, kb_token, attribute_token, blip_token, blip2_token, ofa_token) in enumerate(tqdm(train_loader)):
            v = v.to(device)
            b = b.to(device)
            q = q.to(device)
            a = a.to(device)
            answers = a

            kb_token = kb_token.to(device)
            blip_token = blip_token.to(device)
            blip2_token = blip2_token.to(device)
            ofa_token = ofa_token.to(device)
            attribute_token = attribute_token.to(device)




            lxmert_logit, ans_logit, ans_loss = model(v, b, q, kb_token, attribute_token, blip_token, blip2_token, ofa_token)

            loss_1 = instance_bce_with_logits(lxmert_logit, answers)

            loss_3 = instance_bce_with_logits(ans_logit, answers)

         

            # Total loss
            loss = loss_1 + loss_3 + ans_loss



            print('loss_1',loss_1)
            print('loss_3',loss_3)
            print('ans_loss',ans_loss)



            loss.backward()
            total_norm += nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            num_updates +=1

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        Loss_1/= num_updates
        Loss_3/= num_updates
        Ans_loss/= num_updates

        if eval_loader is not None:
            print("Evaluating...")
            model.train(False)
            cfrf_score, ens_score, ban_lx_score = evaluate(model, eval_loader, device)
            model.train(True)

        logger.write('epoch %d, time: %.2f' % (epoch, time.time()-t))
        logger.write('\Loss_1: %.2f, Loss_3: %.2f' %
                        (Loss_1, Loss_3))
        if eval_loader is not None:
            logger.write('con_score: %.2f, lxmert_score: %.2f,  ans_score: %.2f' % (100 * cfrf_score, 100 * ens_score, 100 * ban_lx_score))

        # Save per epoch
        # if epoch >= saving_epoch:
            # torch.save(model.state_dict(), output + 'model_for_epoch_%d.pth' % epoch)                

            # model_path = os.path.join(output, 'model_epoch%d.pth' % epoch)
            # utils.save_model(model_path, model, epoch, optimizer)
            # Save best epoch
            if eval_loader is not None and ens_score > best_eval_score:
                torch.save(model.state_dict(), os.path.join(output, 'model_epoch%d.pth' % epoch))
                # model_path = os.path.join(output, 'model_epoch_best.pth')
                # utils.save_model(model_path, model, epoch, optimizer)
                best_eval_score = ens_score

def evaluate(model, dataloader, device):
    cfrf_score = 0
    ens_score = 0
    ban_lx_score = 0
    with torch.no_grad():
        for i, (v, b, q, a, qid, kb_token, attribute_token, blip_token, blip2_token, ofa_token) in enumerate(tqdm(dataloader)):
            v = v.to(device)
            b = b.to(device)
            q = q.to(device)
            a = a.to(device)
            kb_token = kb_token.to(device)
            blip_token = blip_token.to(device)
            blip2_token = blip2_token.to(device)
            ofa_token = ofa_token.to(device)
            attribute_token = attribute_token.to(device)

            lxmert_logit, ans_logit, _ = model(v, b, q, kb_token, attribute_token, blip_token, blip2_token, ofa_token)

            ens_score += compute_score_with_logits(lxmert_logit, a).sum()
            cfrf_score += compute_score_with_logits(ans_logit, a).sum()
            logit = ans_logit + lxmert_logit
            ban_lx_score += compute_score_with_logits(logit, a).sum()




    ban_lx_score = ban_lx_score/len(dataloader.dataset)
    cfrf_score = cfrf_score / len(dataloader.dataset)
    ens_score = ens_score / len(dataloader.dataset)  
    
    return cfrf_score, ens_score, ban_lx_score
                


    

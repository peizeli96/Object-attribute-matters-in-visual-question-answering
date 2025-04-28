import os
import time
import torch
import utils as utils
import torch.nn as nn
from transformers import get_linear_schedule_with_warmup
from transformers import AdamW
warmup_updates = 4000
from tqdm import tqdm
from torch.nn import functional as F

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

def train(opt, model, train_loader, eval_loader, num_epochs, output, s_epoch=0):

    grad_clip = opt.clip_norm
    utils.create_dir(output)
    batch_per_epoch = int(len(train_loader.dataset) / opt.batch_size) + 1
    t_total = int(batch_per_epoch * opt.epochs)
    optimizer = torch.optim.Adamax(model.parameters(), lr=opt.lxmert_lr)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0.01 * t_total, num_training_steps=t_total)


    for epoch in range(s_epoch, num_epochs):
        total_norm = 0
        num_updates = 0
        Loss_1 = 0 
        Loss_3 = 0 

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


            loss_1 = instance_bce(lxmert_logit, answers)

            loss_3 = instance_bce(ans_logit, answers)            
                     

            # Total loss
            loss = loss_1 + loss_3 + ans_loss


            loss.backward()
            total_norm += nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            num_updates +=1

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()


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

         
            if eval_loader is not None and ens_score > best_eval_score:
                torch.save(model.state_dict(), os.path.join(output, 'model_epoch%d.pth' % epoch))                
                best_eval_score = ens_score


def evaluate(model, dataloader, device):
    score = 0
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

            lxmert_logit, ans_logit,_ = model(v, b, q, kb_token, attribute_token, blip_token, blip2_token, ofa_token)
            score += compute_score_with_logits(ans_logit, a).sum()
    score = score / len(dataloader.dataset)      
    return score

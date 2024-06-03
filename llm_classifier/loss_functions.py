import torch


def compute_minibatch_bce_loss(logits, ranks):
    logits = logits.float()
    ranks = ranks.float()
    logits_expanded = logits.repeat(1, logits.size(0))
    ranks_expanded = ranks.unsqueeze(1).repeat(1, ranks.size(0))
    score_diff = logits_expanded - logits_expanded.t()
    target = (ranks_expanded < ranks_expanded.t()).float()
    sigmoid_diff = torch.sigmoid(score_diff)
    loss_ij = -target * torch.log(sigmoid_diff) - (1 - target) * torch.log(1 - sigmoid_diff)
    mask = torch.triu(torch.ones_like(loss_ij), diagonal=1)
    loss_ij *= mask
    total_loss = loss_ij.sum()
    return total_loss


def compute_minibatch_hinge_loss(logits, ranks):
    logits = logits.float()
    ranks = ranks.float()
    logits_expanded = logits.repeat(1, logits.size(0))
    ranks_expanded = ranks.unsqueeze(1).repeat(1, ranks.size(0))
    score_diff = logits_expanded - logits_expanded.t()
    margin = ranks_expanded.t() - ranks_expanded
    target = (ranks_expanded < ranks_expanded.t()).float()
    loss_ij = torch.clamp(margin - score_diff, min=0)
    loss_ij *= target
    total_loss = loss_ij.sum()
    return total_loss


def compute_minibatch_bce_loss_same_rank_handling(logits, ranks):
    logits = logits.float()
    ranks = ranks.float()
    logits_expanded = logits.repeat(1, logits.size(0))
    ranks_expanded = ranks.unsqueeze(1).repeat(1, ranks.size(0))
    score_diff = logits_expanded - logits_expanded.t()
    target = (ranks_expanded < ranks_expanded.t()).float()

    # Forcing same ranked elements to have close scores
    same_rank_mask = (ranks_expanded == ranks_expanded.t()).float()
    same_rank_loss = (score_diff ** 2) * same_rank_mask

    sigmoid_diff = torch.sigmoid(score_diff)
    ranking_loss_ij = -target * torch.log(sigmoid_diff) - (1 - target) * torch.log(1 - sigmoid_diff)

    # Ignoring the case where ranks are equal in the ranking loss
    ranking_loss_ij *= (1 - same_rank_mask)

    mask = torch.triu(torch.ones_like(ranking_loss_ij), diagonal=1)
    ranking_loss_ij *= mask
    same_rank_loss *= mask
    total_ranking_loss = ranking_loss_ij.sum()
    total_same_rank_loss = same_rank_loss.sum()

    # Combine the ranking loss and the same-rank closeness loss
    total_loss = total_ranking_loss + total_same_rank_loss
    return total_loss


def compute_minibatch_loss(logits, ranks, loss_name):
    if loss_name == 'bce':
        return compute_minibatch_bce_loss(logits, ranks)
    elif loss_name == 'bce_same_rank_handling':
        return compute_minibatch_bce_loss_same_rank_handling(logits, ranks)
    elif loss_name == 'hinge':
        return compute_minibatch_hinge_loss(logits, ranks)
    else:
        raise Exception('Loss name incorrect')

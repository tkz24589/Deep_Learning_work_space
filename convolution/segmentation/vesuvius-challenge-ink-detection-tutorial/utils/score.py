def dice_coef(targets, preds, thr=0.5, beta=0.5, smooth=1e-5):

    #comment out if your model contains a sigmoid or equivalent activation layer
    # flatten label and prediction tensors
    preds = (preds > thr).view(-1).float()
    targets = targets.view(-1).float()

    y_true_count = targets.sum()
    ctp = preds[targets==1].sum()
    cfp = preds[targets==0].sum()
    beta_squared = beta * beta

    c_precision = ctp / (ctp + cfp + smooth)
    c_recall = ctp / (y_true_count + smooth)
    dice = (1 + beta_squared) * (c_precision * c_recall) / (beta_squared * c_precision + c_recall + smooth)

    return dice
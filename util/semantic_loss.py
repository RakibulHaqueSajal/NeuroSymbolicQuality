import torch
import torch.nn.functional as F
import torch

import torch.nn as nn
import torch.nn.functional as F
def corn_probs(logits: torch.Tensor) -> torch.Tensor:
    """
    Convert CORN logits [B, K-1] to class probabilities [B, K], K=4 here.
    """
    p_gt = torch.sigmoid(logits)          # [B, 3] => P(y>1), P(y>2), P(y>3)
    p1 = 1.0 - p_gt[:, 0]
    p2 = p_gt[:, 0] - p_gt[:, 1]
    p3 = p_gt[:, 1] - p_gt[:, 2]
    p4 = p_gt[:, 2]
    probs = torch.stack([p1, p2, p3, p4], dim=-1)  # [B, 4]
    return torch.clamp(probs, 1e-6, 1.0)           # avoid exact 0


# def corn_probs(logits: torch.Tensor) -> torch.Tensor:
#     cond = torch.sigmoid(logits)           # [B,3] = c0,c1,c2
#     c0, c1, c2 = cond[:,0], cond[:,1], cond[:,2]

#     p_gt0 = c0               # P(y>0)
#     p_gt1 = c0 * c1          # P(y>1)
#     p_gt2 = c0 * c1 * c2     # P(y>2)

#     p0 = 1.0 - p_gt0
#     p1 = p_gt0 - p_gt1
#     p2 = p_gt1 - p_gt2
#     p3 = p_gt2

#     probs = torch.stack([p0, p1, p2, p3], dim=-1)
#     probs = torch.clamp(probs, 1e-6, 1.0)
#     probs = probs / probs.sum(dim=-1, keepdim=True)
#     return probs


##Lukasiewicz

def implication_loss(p_ante: torch.Tensor, p_cons: torch.Tensor, reduction="mean"):
    """
    p_ante, p_cons: [B] truth degrees in [0,1].
    Returns small value if implication mostly satisfied.
    """
    t_imp = torch.clamp(1.0 - p_ante + p_cons, 0.0, 1.0)  # [B]
    loss = 1.0 - t_imp
    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    else:
        return loss

#RuleBased Semantic Loss

def rule_based_semantic_loss(main_logits, c1_logits, c2_logits, c3_logits,
                             weights=None) -> torch.Tensor:
    """
    Rule-based semantic loss using fuzzy implications.
    All logits are CORN logits [B, 3] for 4 ordinal classes.
    """

    # Convert logits -> class probabilities [B,4]
    q_probs = corn_probs(main_logits)  # overall quality
    s_probs = corn_probs(c1_logits)    # sharpness
    n_probs = corn_probs(c2_logits)    # myocardium nulling
    e_probs = corn_probs(c3_logits)    # enhancement (aorta & valves)

    # Convenience aliases: P(score=k)
    q1, q2, q3, q4 = [q_probs[:, i] for i in range(4)]
    s1, s2, s3, s4 = [s_probs[:, i] for i in range(4)]
    n1, n2, n3, n4 = [n_probs[:, i] for i in range(4)]
    e1, e2, e3, e4 = [e_probs[:, i] for i in range(4)]

    # ---- Rule 1: if any factor is 1 → quality is 1 ----
    any_factor_1 = 1.0 - (1.0 - s1) * (1.0 - n1) * (1.0 - e1)
    r1_loss = implication_loss(any_factor_1, q1)

    # ---- Rule 2: if no factor is 1 and some factor is 2 → quality is 2 ----
    no_factor_1 = (1.0 - s1) * (1.0 - n1) * (1.0 - e1)
    any_factor_2 = 1.0 - (1.0 - s2) * (1.0 - n2) * (1.0 - e2)
    antecedent2 = no_factor_1 * any_factor_2
    r2_loss = implication_loss(antecedent2, q2)

    # ---- Rule 3: if E=3 and S,N >=3 → quality is 3 ----
    s_ge3 = s3 + s4
    n_ge3 = n3 + n4
    antecedent3 = e3 * s_ge3 * n_ge3
    r3_loss = implication_loss(antecedent3, q3)

    # ---- Rule 4: if all factors are 4 → quality is 4 ----
    antecedent4 = s4 * n4 * e4
    r4_loss = implication_loss(antecedent4, q4)

    if weights is None:
        weights = [1.0, 1.0, 1.0, 1.0]

    total = (weights[0] * r1_loss +
             weights[1] * r2_loss +
             weights[2] * r3_loss +
             weights[3] * r4_loss)

    return total




#t-norms and t-conorms with product

def tnorm_product(*args: torch.Tensor) -> torch.Tensor:
    """
    Product T-norm: AND over args (all same shape).
    """
    out = torch.ones_like(args[0])
    for a in args:
        out = out * a
    return out


def tconorm_product(*args: torch.Tensor) -> torch.Tensor:
    """
    Product T-conorm: OR over args (all same shape).
    """
    out = torch.ones_like(args[0])
    for a in args:
        out = out * (1.0 - a)
    return 1.0 - out

def implication_loss_product(p_ante: torch.Tensor,
                             p_cons: torch.Tensor,
                             reduction: str = "mean",
                             eps: float = 1e-6) -> torch.Tensor:
    """
    Product fuzzy implication loss for 'IF A THEN B':
        I_P(A,B) = 1        if A <= B
                   B / A    if A > B
    Loss = 1 - I_P(A,B).
    p_ante, p_cons: [B] truth degrees in [0,1].
    """
    # ratio B/A for violating region
    ratio = p_cons / (p_ante + eps)
    imp = torch.where(p_ante <= p_cons,
                      torch.ones_like(p_ante),
                      ratio)
    imp = torch.clamp(imp, 0.0, 1.0)
    loss = 1.0 - imp       # [B]

    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    else:
        return loss


def rule_based_semantic_loss_product(
    main_logits: torch.Tensor,
    c1_logits: torch.Tensor,
    c2_logits: torch.Tensor,
    c3_logits: torch.Tensor,
    weights=None
) -> torch.Tensor:
    """
    Rule-based semantic loss using:
      - Product T-norm for AND
      - Product T-conorm for OR
      - Product fuzzy implication for IF-THEN
    All logits: CORN logits [B, 3] for 4 ordinal classes.
    main_logits: overall quality
    c1_logits: sharpness
    c2_logits: myocardium nulling
    c3_logits: enhancement of aorta & valves
    """

    # ---- 1) logits -> probabilities [B,4] ----
    q_probs = corn_probs(main_logits)
    s_probs = corn_probs(c1_logits)
    n_probs = corn_probs(c2_logits)
    e_probs = corn_probs(c3_logits)




    # Unpack per-class probabilities
    q1, q2, q3, q4 = [q_probs[:, i] for i in range(4)]
    s1, s2, s3, s4 = [s_probs[:, i] for i in range(4)]
    n1, n2, n3, n4 = [n_probs[:, i] for i in range(4)]
    e1, e2, e3, e4 = [e_probs[:, i] for i in range(4)]

    # ---- Rule 1: if any factor is 1 -> quality is 1 ----
    any_factor_1 = tconorm_product(s1, n1, e1)          # OR via product t-conorm
    r1_loss = implication_loss_product(any_factor_1, q1)

    # ---- Rule 2: if no factor is 1 and some factor is 2 -> quality is 2 ----
    no_factor_1 = tnorm_product(1.0 - s1, 1.0 - n1, 1.0 - e1)  # AND via product t-norm
    any_factor_2 = tconorm_product(s2, n2, e2)
    antecedent2 = no_factor_1 * any_factor_2
    r2_loss = implication_loss_product(antecedent2, q2)

    # ---- Rule 3: if E=3 and S,N>=3 -> quality is 3 ----
    s_ge3 = s3 + s4
    n_ge3 = n3 + n4
    antecedent3 = tnorm_product(e3, s_ge3, n_ge3)
    r3_loss = implication_loss_product(antecedent3, q3)

    # ---- Rule 4: if all factors are 4 -> quality is 4 ----
    antecedent4 = tnorm_product(s4, n4, e4)
    r4_loss = implication_loss_product(antecedent4, q4)

    if weights is None:
        weights = [1.0, 1.0, 1.0, 1.0]

    total = (
        weights[0] * r1_loss +
        weights[1] * r2_loss +
        weights[2] * r3_loss +
        weights[3] * r4_loss
    )
    return total


#Making the Weights of the Rules Leanarable as well


def rule_losses_product(
    main_logits: torch.Tensor,
    c1_logits: torch.Tensor,
    c2_logits: torch.Tensor,
    c3_logits: torch.Tensor,
) -> torch.Tensor:
    """
    Compute per-rule semantic losses (without weighting).
    Returns: tensor [4] = [r1_loss, r2_loss, r3_loss, r4_loss]
    """

    # ---- 1) logits -> probabilities [B,4] ----
    q_probs = corn_probs(main_logits)   # overall quality
    s_probs = corn_probs(c1_logits)     # sharpness
    n_probs = corn_probs(c2_logits)     # myocardium nulling
    e_probs = corn_probs(c3_logits)     # enhancement

    # Unpack per-class probabilities
    q1, q2, q3, q4 = [q_probs[:, i] for i in range(4)]
    s1, s2, s3, s4 = [s_probs[:, i] for i in range(4)]
    n1, n2, n3, n4 = [n_probs[:, i] for i in range(4)]
    e1, e2, e3, e4 = [e_probs[:, i] for i in range(4)]

    # ---- Rule 1: if any factor is 1 -> quality is 1 ----
    any_factor_1 = tconorm_product(s1, n1, e1)
    r1_loss = implication_loss_product(any_factor_1, q1, reduction="mean")

    # ---- Rule 2: if no factor is 1 and some factor is 2 -> quality is 2 ----
    no_factor_1 = tnorm_product(1.0 - s1, 1.0 - n1, 1.0 - e1)
    any_factor_2 = tconorm_product(s2, n2, e2)
    antecedent2 = no_factor_1 * any_factor_2
    r2_loss = implication_loss_product(antecedent2, q2, reduction="mean")

    # ---- Rule 3: if E=3 and S,N>=3 -> quality is 3 ----
    s_ge3 = s3 + s4
    n_ge3 = n3 + n4
    antecedent3 = tnorm_product(e3, s_ge3, n_ge3)
    r3_loss = implication_loss_product(antecedent3, q3, reduction="mean")

    # ---- Rule 4: if all factors are 4 -> quality is 4 ----
    antecedent4 = tnorm_product(s4, n4, e4)
    r4_loss = implication_loss_product(antecedent4, q4, reduction="mean")

    return torch.stack([r1_loss, r2_loss, r3_loss, r4_loss], dim=0)  # [4]




class RuleSemanticLoss(nn.Module):
    def __init__(self, init_weights=None, normalize="softmax"):
        """
        Learnable weights over 4 semantic rules.

        normalize="softmax": weights are positive and sum to 1.
        normalize="softplus": weights are positive but unconstrained sum.
        """
        super().__init__()
        if init_weights is None:
            init_weights = torch.ones(4)  # start roughly equal

        # store logits for weights so softmax doesn't start saturated
        init_logits = init_weights.log()
        self.rule_logits = nn.Parameter(init_logits)  # [4]
        self.normalize = normalize

    def forward(self, main_logits, c1_logits, c2_logits, c3_logits):
        # per-rule losses [4]
        rule_losses = rule_losses_product(main_logits, c1_logits, c2_logits, c3_logits)

        # convert logits -> weights
        if self.normalize == "softmax":
            weights = F.softmax(self.rule_logits, dim=0)  # sum to 1
        elif self.normalize == "softplus":
            weights = F.softplus(self.rule_logits)        # >0, unconstrained sum
        else:
            raise ValueError(f"Unknown normalize mode: {self.normalize}")

        total = (weights * rule_losses).sum()
        return total, rule_losses.detach(), weights.detach()
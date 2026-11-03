"""Train CLUB on the paper's Gaussian schedule at several q(y|x) capacities.

Schedule matches the upstream notebook (mi_estimation.ipynb): d=20, batch 64,
Adam lr 0.005, 4000 steps per MI level, MI levels 2,4,6,8,10 trained in sequence,
estimate = mean of the last 500 per-step estimates at each level.
"""
import argparse, json, math, time
import torch
from club import CLUB, CLUBSample
from gaussian import mi_to_rho, true_mi, oracle_club, sample

MI_LEVELS = [2.0, 4.0, 6.0, 8.0, 10.0]
DIM, BATCH, LR, STEPS = 20, 64, 0.005, 4000


def run(model_cls, hidden, seed, steps=STEPS):
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    model = model_cls(DIM, DIM, hidden)
    opt = torch.optim.Adam(model.parameters(), LR)
    rows = []
    for mi in MI_LEVELS:
        rho = mi_to_rho(mi, DIM)
        ests = []
        for _ in range(steps):
            x, y = sample(rho, DIM, BATCH, g)
            model.eval()
            with torch.no_grad():
                ests.append(model(x, y).item())
            model.train()
            loss = model.learning_loss(x, y)
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            x, y = sample(rho, DIM, 4096, g)
            mu, logvar = model.get_mu_logvar(x)
            # average Gaussian log-likelihood per sample of q vs the true conditional
            ll_q = (-0.5 * ((y - mu) ** 2 / logvar.exp() + logvar + math.log(2 * math.pi))).sum(1).mean().item()
            tv = 1 - rho ** 2
            ll_p = (-0.5 * ((y - rho * x) ** 2 / tv + math.log(tv) + math.log(2 * math.pi))).sum(1).mean().item()
        last = ests[-500:]
        rows.append({
            "true_mi": mi,
            "estimate": sum(last) / len(last),
            "oracle_club": oracle_club(rho, DIM),
            "q_loglik_gap": ll_p - ll_q,  # KL-ish: how far q is from p(y|x), in nats
        })
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--out", default="results.json")
    args = ap.parse_args()
    configs = [("CLUB", CLUB, 15), ("CLUB", CLUB, 64), ("CLUB", CLUB, 256), ("CLUB", CLUB, None),
               ("CLUBSample", CLUBSample, 15), ("CLUBSample", CLUBSample, 256)]
    out = []
    t0 = time.time()
    for name, cls, hidden in configs:
        for seed in range(args.seeds):
            for row in run(cls, hidden, seed):
                out.append({"model": name, "hidden_size": hidden, "seed": seed, **row})
            print(f"{name} hidden={hidden} seed={seed} done ({time.time() - t0:.0f}s)", flush=True)
    json.dump(out, open(args.out, "w"), indent=1)

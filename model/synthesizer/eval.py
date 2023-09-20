import torch


def _js_divergence(p, q):
    eps = 1e-8
    m = 0.5 * (p + q)
    return 0.5 * (p * ((p / m + eps).log()) + q * ((q / m + eps).log())).sum()


def get_distribution(data, bins=20):
    hist = torch.histc(data, bins=bins, min=0, max=1)
    return hist / hist.sum()


def get_jsd(data_p, data_q):
    p = get_distribution(data_p)
    q = get_distribution(data_q)
    return _js_divergence(p, q)

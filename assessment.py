# assessment.py

import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim
from sentence_transformers import SentenceTransformer
import torch
from torchvision.transforms import Resize
from torchvision.models import inception_v3
from scipy.linalg import sqrtm
from PIL import Image


def temporal_consistency(video):
    """
    Calculate temporal consistency using frame-to-frame SSIM.
    """
    frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in video]
    scores = []
    for i in range(1, len(frames)):
        score, _ = ssim(frames[i - 1], frames[i], full=True)
        scores.append(score)
    return np.mean(scores)


def inception_score(video, model):
    """
    Calculate Inception Score for video frames.
    """
    preds = []
    for frame in video:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Resize((299, 299))(torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0).unsqueeze(0)
        pred = torch.nn.functional.softmax(model(frame), dim=1).detach().cpu().numpy()
        preds.append(pred)
    preds = np.concatenate(preds, axis=0)
    scores = []
    for i in range(preds.shape[0]):
        part = preds[[j for j in range(preds.shape[0]) if j != i]]
        kl = part * (np.log(part) - np.log(np.mean(part, axis=0, keepdims=True)))
        kl = np.mean(np.sum(kl, axis=1))
        scores.append(np.exp(kl))
    return np.mean(scores)


def prompt_similarity(video, prompt, model):
    """
    Calculate similarity between video frames and the prompt.
    """
    prompt_embedding = model.encode(prompt)
    similarities = []
    for frame in video:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_embedding = model.encode(Image.fromarray(frame_rgb))
        similarity = np.dot(prompt_embedding, frame_embedding) / (
                    np.linalg.norm(prompt_embedding) * np.linalg.norm(frame_embedding))
        similarities.append(similarity)
    return np.mean(similarities)


def fid_score(video1, video2, model):
    """
    Calculate FID score between two videos.
    """

    def calculate_statistics(video):
        features = []
        for frame in video:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Resize((299, 299))(torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0).unsqueeze(0)
            feature = model(frame).detach().cpu().numpy()
            features.append(feature)
        features = np.concatenate(features, axis=0)
        mu = np.mean(features, axis=0)
        sigma = np.cov(features, rowvar=False)
        return mu, sigma

    mu1, sigma1 = calculate_statistics(video1)
    mu2, sigma2 = calculate_statistics(video2)

    diff = mu1 - mu2
    covmean = sqrtm(sigma1.dot(sigma2))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
    return fid
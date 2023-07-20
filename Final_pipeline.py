from VAE_architecture import VAE, AE
import torch
import torchvision.transforms as T
from PIL import Image
from transformers import BertTokenizer, BertModel
import pandas as pd
import numpy as np
from torchvision.utils import save_image

def model_inference(
    Ca_ion_mM,
    CO3_ion_mM,
    HCO3_ion_mM,
    Polymer_type,
    Polymer_Mwt,
    Polymer_pwt,
    Surfactant_type,
    Surfactant_pwt,
    Solvent_type,
    Solvent_pvol,
    Stirring_rpm,
    Temperature_C,
    Time_sec,
):
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased")
    ae = AE.load_from_checkpoint("data/Checkpoints/ae.ckpt").eval()
    vae = VAE.load_from_checkpoint("data/Checkpoints/vae.ckpt").eval()
    template = "All the materials were synthesized by the co-precipitation technique. First, 1 M CaCl2 with final concentration of Ca ion being {} mM was mixed with {} polymer with molecular weight of {} kDa and content being {} % wt. Then, {} solvent was added with final volume content being {}, following adjustment with distilled water up to 500 mkl. Then, 0.1 M Na2CO3 with final concentration of CO3 ions being {} was mixed with 0.1 M of NaHCO3 with final concentration of HCO3 ions being {} and {} surfactant with content being {} % wt. Then, same solvent was added, following adjustment with distilled water up to 500 mkl. Two resulting solutions, heated up to {} C before the reaction, were mixed under the stirring with {} rpm, while the temperature kept unchanged. Reaction proceeded for {} sec following centrifugation."
    formatted_template = template.format(
        Ca_ion_mM,
        Polymer_type,
        Polymer_Mwt,
        Polymer_pwt,
        Solvent_type,
        Solvent_pvol,
        CO3_ion_mM,
        HCO3_ion_mM,
        Surfactant_type,
        Surfactant_pwt,
        Temperature_C,
        Stirring_rpm,
        Time_sec,
    )
    with torch.no_grad():
        encoded_inputs = tokenizer(
            formatted_template,
            return_tensors="pt",
            padding="max_length",
            max_length=250,
        )
        outputs = model(**encoded_inputs).last_hidden_state[:, 0, :]
        image_embeddings = ae(outputs.to(ae.device))
        mu = vae.fc_mu(image_embeddings.to(vae.device))
        log_var = vae.fc_var(image_embeddings.to(vae.device))
        p, q, z = vae.sample(mu, log_var)
        image = vae.decoder(z)
    return formatted_template, image


def model_inference_old(input):
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased")
    ae = AE.load_from_checkpoint("data/Checkpoints/ae.ckpt").eval()
    vae = VAE.load_from_checkpoint("data/Checkpoints/vae.ckpt").eval()
    template = "All the materials were synthesized by the co-precipitation technique. First, 1 M CaCl2 with final concentration of Ca ion being {} mM was mixed with {} polymer with molecular weight of {} kDa and content being {} % wt. Then, {} solvent was added with final volume content being {}, following adjustment with distilled water up to 500 mkl. Then, 0.1 M Na2CO3 with final concentration of CO3 ions being {} was mixed with 0.1 M of NaHCO3 with final concentration of HCO3 ions being {} and {} surfactant with content being {} % wt. Then, same solvent was added, following adjustment with distilled water up to 500 mkl. Two resulting solutions, heated up to {} C before the reaction, were mixed under the stirring with {} rpm, while the temperature kept unchanged. Reaction proceeded for {} sec following centrifugation."
    formatted_template = template.format(
        input[0],
        input[3],
        input[4],
        input[5],
        input[-5],
        input[-4],
        input[1],
        input[2],
        input[6],
        input[7],
        input[-2],
        input[-3],
        time_to_sec(input[-1]),
    )
    with torch.no_grad():
        encoded_inputs = tokenizer(
            formatted_template,
            return_tensors="pt",
            padding="max_length",
            max_length=250,
        )
        outputs = model(**encoded_inputs).last_hidden_state[:, 0, :]
        image_embeddings = ae(outputs.to(ae.device))
        mu = vae.fc_mu(image_embeddings.to(vae.device))
        log_var = vae.fc_var(image_embeddings.to(vae.device))
        p, q, z = vae.sample(mu, log_var)
        image = vae.decoder(z)
    return image

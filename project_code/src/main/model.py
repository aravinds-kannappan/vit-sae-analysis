import torch

def predict(model, dataloader, source, RPI= False, magnitude = 1.0, half = True, verbose = False):
  device = "cuda" if torch.cuda.is_available() else "cpu"

  # Attach hook to apply RPI intervention

  if RPI:
    def RPI_hook(module, input, output):
      B, C, H, W = output.shape
      out = output.view(B, C, -1).contiguous()

      perm = torch.randperm(out.shape[-1])
      return out[:,:,perm].view(B,C,H,W)

    # Locate the patch embedding conv robustly (attribute names differ across
    # transformers versions and between transformers and timm).
    import os, sys
    sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/.."))
    from main.load_models import get_patch_embed_conv

    handle = get_patch_embed_conv(model, source).register_forward_hook(RPI_hook)

  # Scale positional encodings (for the PE magnitude scaling experiment)
  if source == "transformers":
    try:
      original_pe = model._modules['vit'].embeddings.position_embeddings
      model._modules['vit'].embeddings.position_embeddings = torch.nn.Parameter(model._modules['vit'].embeddings.position_embeddings * magnitude)
    except:
      pass
  elif source == "timm":
    pass
  acc_list = [] # List of accuracies

  model.eval()
  # Half precision is only used on CUDA. On CPU/MPS fp16 forward is unsupported or
  # very slow, so keep float32 there. The Data class gates input casting the same
  # way, so model and inputs always share a dtype.
  if half and torch.cuda.is_available():
    model = model.half()
    
  with torch.inference_mode():
    for images, labels in dataloader:
      images = images.to(device)
      if source == "transformers":
        outputs = model(**images)
        logits = outputs.logits
      elif source == "timm":
        logits = model(images)
      
      predicted_class_idx = logits.argmax(-1).to(device)
      accuracy = (predicted_class_idx == torch.as_tensor(labels).to(device)).sum() / len(labels)
      accuracy = accuracy.detach().cpu().item()
      acc_list.append(accuracy)
      if verbose:
        print(accuracy)
  
  #Resetting the model's PEs to their original magnitude
  try:
    model._modules['vit'].embeddings.position_embeddings = original_pe
  except:
    pass

  if RPI:
    handle.remove()

  return sum(acc_list) / len(acc_list)
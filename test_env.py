import torch
from samgeo import SamGeo2

sam = SamGeo2(
    model_id="sam2-hiera-large",
    automatic=False, 
)
print(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
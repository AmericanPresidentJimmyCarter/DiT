import torch
from unclip_prior import UnCLIPPriorPipeline
pl = UnCLIPPriorPipeline.from_pretrained("kakaobrain/karlo-v1-alpha", torch_dtype=torch.float32)
foo = pl(['foo', 'bar', 'car', 'train'])

if foo.size() == (4, 768):
    print('OK')
else:
    print('bad output', foo.size())

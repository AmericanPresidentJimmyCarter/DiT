from oclip_prior import load_prior_model, image_embeddings_for_text

prior = load_prior_model()
foo = image_embeddings_for_text(prior, ['foo', 'bar'])
assert foo.size() == (2, 1024)

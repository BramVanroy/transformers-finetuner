TODO
====

- make sure that tokenized dataset is cached: seems very hard to do when using OO because tokenize func needs to be picklable
- optimize hypopt PB2; make sure it works -- on hold for now
- automatically build model card
- push to hub utility function
- figure out how to get the best checkpoint from gridsearch, so that we do not have to train again
- investigate TQDM issues when using ray gridsearch on Windows?
- make sure data loading/preprocesing works better in multi-gpu systems: to do so, we need to do data processing once 
in the main process and make sure it caches and then use the cached version in other processes. But currently we get a 
lot of caching/fingerprint issues because it cannot pickle everything.
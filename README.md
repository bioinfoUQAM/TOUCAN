# **TOUCAN: Supervised learning for fungal BGC discovery**

A supervised learning framework to predict Biosynthetic Gene Clusters (BGCs) in fungi based on a combination of feature types (k-mers, Pfam protein domains, and GO terms). 

## **How to: Classification**

Make a copy of `/src/config.init.DEFAULT`, and rename it to `/src/config.init`. Update the `[default] home` to the current project root path.

### **Configure**

At the `[prediction]` section  in the `config.init` file, specify the minimum parameters accordingly:

- the `task`: `train`, `validation`, or `test` 
- indicate the corpus location in `source.path`
- (if using sequences) indicate the `source.type`: `nucleotide` or `aminoacid`
- specify the positive instances % in `pos.perc`
- indicate the `feat.type` as `kmers`, `domains` or `go` (if combining multiple features, separate them with a `-`, as in `go-kmers-domains`)
- set the minimum occurrences to consider a feature in `feat.minOcc`
- set the k-mer length in `feat.size`
- select a `classifier`: `logit`, `mlp`, `linearsvc`, `nusvc`, `svc`, `randomforest`

### **Run**

To run the classification task from the project `virtualenv` simply:

```bash
(.env) user@foo:~fungalbgcs/src$ python -m pipeprediction.ML
```

### **Output**

The `train` task will generate a `/metrics` folder, with:

- the (re-load-able) model file `(classifier)_(featuretype).model.pkl`
- a list of features file `(featuretype).feat`

The `validation` task will also generate in the `/metrics` folder:

- a performance file `(classifier)_(featuretype).valid` with P, R, F-m and a confusion matrix
- a list of *{valid_instance_IDs, predicted label}* file `(classifier)_(featuretype).IDs.valid`

The `test` task requires either `train` or `validation` to have been performed, since it will read from the model `*.model.pkl` and feature `*.feat` files. It generates in the `/metrics` folder:

- a performance file `(classifier)_(featuretype).test` with P, R, F-m and a confusion matrix
- a list of *{test_instance_IDs, predicted label}* file `(classifier)_(featuretype)_(testfolder).IDs.test`, used as input for evaluation against gold clusters

## **Resources**

**Datasets:** Openly available [fungal BGC datasets](https://github.com/bioinfoUQAM/fungalbgcdata) to train and validate models (details [here](https://arxiv.org/abs/2001.03260)).

**External software:** To set up Pfam for protein domain annotation locally, please refer to the steps on `/extSoftware/`.



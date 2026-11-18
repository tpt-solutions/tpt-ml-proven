# Changelog

All notable changes to `tpt-ml-proven-ensemble` are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- `StumpForest`: bagged depth-one classification trees with majority vote and Gini splits.
  No scikit-learn reference, so it is tested through its properties. Contract:
  `docs/contracts/stump-forest.tptc`.

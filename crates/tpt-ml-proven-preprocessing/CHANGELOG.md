# Changelog

All notable changes to `tpt-ml-proven-preprocessing` are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- `StandardScaler`, `MinMaxScaler` and `RobustScaler`, with `inverse_transform`.
- `OneHotEncoder` and `OrdinalEncoder`. Unknown categories are an error.
- `SimpleImputer` with mean, median, most-frequent and constant strategies.
- Parity tests (1e-9) for the three scalers against scikit-learn on a committed fixture.

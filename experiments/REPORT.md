# Experiment Report

| name | final_equity | cum_return_% | sharpe | max_drawdown | trades |
|---|---:|---:|---:|---:|---:|
| exp_ev_arbiter_top10 | 199349.7417890101 | 99.35 | 1.8450521746 | -0.1352710352 | 1157 |
| exp_ev_arbiter_top15 | 192769.0704883231 | 92.77 | 1.9516731368 | -0.148545818 | 1167 |
| exp_ev_arbiter_top20 | 192769.0704883231 | 92.77 | 1.9516731368 | -0.148545818 | 1167 |
| exp_optimizer_top20 | 229804.9533666135 | 129.80 | 1.5786984246 | -0.1498903848 | 1342 |
| exp_events_ev_arbiter_top15 | 211650.3789409724 | 111.65 | 1.8727711841 | -0.1414070333 | 1228 |
| exp_alphas_optimizer_top20 | 366854.3315297296 | 266.85 | 1.9290251264 | -0.1338613288 | 1705 |
| exp_consensus_optimizer_top20 | 181005.3371576868 | 81.01 | 0.9188487297 | -0.2039371233 | 1704 |
| exp_no_rl_arbiter_top15 | 196118.5212036712 | 96.12 | 1.8131840544 | -0.1335216732 | 1166 |
| exp_visuals_focus_top15 | 237974.2198271583 | 137.97 | 1.5299686502 | -0.1571438752 | 988 |
| exp_cn_consensus_top20 | 237974.2198271583 | 137.97 | 1.5299686502 | -0.1571438752 | 988 |

## Experiment Details
### exp_ev_arbiter_top10
- RL+heuristic arbiter; Top-N=10; Lookback=180; Test window: 2024-01-01 to 2025-01-31
- Plots: normalized_returns, rolling_sharpe, turnover, returns_hist, returns_qq, rolling_beta, weights_heatmap, calibration_curve
### exp_ev_arbiter_top15
- RL+heuristic arbiter; Top-N=15; Lookback=180; Test window: 2024-01-01 to 2025-01-31
- Plots: normalized_returns, rolling_sharpe, turnover, returns_hist, returns_qq, rolling_beta, weights_heatmap, calibration_curve
### exp_ev_arbiter_top20
- RL+heuristic arbiter; Top-N=20; Lookback=180; Test window: 2024-01-01 to 2025-01-31
- Plots: normalized_returns, rolling_sharpe, turnover, returns_hist, returns_qq, rolling_beta, weights_heatmap, calibration_curve
### exp_optimizer_top20
- Mean-variance optimizer with shrinkage covariance; Top-N=20; Lookback=180; Test window: 2024-01-01 to 2025-01-31
- Plots: normalized_returns, rolling_sharpe, turnover, returns_hist, returns_qq, rolling_beta, weights_heatmap, calibration_curve
### exp_events_ev_arbiter_top15
- RL+heuristic arbiter with event-driven boosts (gap/volume); Top-N=15; Lookback=180; Test window: 2024-01-01 to 2025-01-31
- Plots: normalized_returns, rolling_sharpe, turnover, returns_hist, returns_qq, rolling_beta, weights_heatmap, calibration_curve
### exp_alphas_optimizer_top20
- Optimizer with alpha-augmented features (Qlib-like); Top-N=20; Lookback=180; Test window: 2024-01-01 to 2025-01-31
- Plots: normalized_returns, rolling_sharpe, turnover, returns_hist, returns_qq, rolling_beta, weights_heatmap, calibration_curve
### exp_consensus_optimizer_top20
- Consensus (router+aggregator) provides mu into mean-variance optimizer; Top-N=20; Lookback=180; Test window: 2024-01-01 to 2025-01-31
- Plots: normalized_returns, rolling_sharpe, turnover, returns_hist, returns_qq, rolling_beta, weights_heatmap, calibration_curve
### exp_no_rl_arbiter_top15
- Heuristic arbiter (no RL); Top-N=15; Lookback=180; Test window: 2024-01-01 to 2025-01-31
- Plots: normalized_returns, rolling_sharpe, turnover, returns_hist, returns_qq, rolling_beta, weights_heatmap, calibration_curve
### exp_visuals_focus_top15
- Mean-variance optimizer with shrinkage covariance; Top-N=15; Lookback=180; Test window: 2024-01-01 to 2025-01-31
- Plots: normalized_returns, rolling_sharpe, turnover, returns_hist, returns_qq, rolling_beta, weights_heatmap, calibration_curve
### exp_cn_consensus_top20
- Consensus (router+aggregator) provides mu into mean-variance optimizer; Top-N=20; Lookback=180; Test window: 2024-01-01 to 2025-01-31
- Plots: normalized_returns, rolling_sharpe, turnover, returns_hist, returns_qq, rolling_beta, weights_heatmap, calibration_curve
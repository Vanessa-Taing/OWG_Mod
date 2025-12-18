# Test Report - 2025-12-18T21:31:49

## Summary

- **Total collected**: 54
- **Passed**: 53
- **Failed**: 0
- **Errors**: 0
- **Skipped**: 1
- **Expected failures (xfailed)**: 0
- **Unexpected passes (xpassed)**: 0
- **Exit status**: 0

## By Test

### Passed (53)

- **tests/test_app_utils.py::test_safe_column_exists_true_and_false**
- **tests/test_app_utils.py::test_clean_confidence_data_replaces_minus_one_with_nan**
- **tests/test_app_utils.py::test_calculate_overall_confidence_uses_mean_across_stages**
- **tests/test_app_utils.py::test_filter_dataframe_basic_filters**
- **tests/test_app_utils.py::test_filter_dataframe_model_category_and_query_and_ranges**
- **tests/test_app_utils.py::test_calculate_success_rate_overall_and_grouped**
- **tests/test_app_utils.py::test_perform_ttest_and_nonparametric**
- **tests/test_app_utils.py::test_perform_anova_and_posthoc**
- **tests/test_app_utils.py::test_calculate_calibration_metrics_basic**
- **tests/test_app_utils.py::test_get_significance_marker[nan-]**
- **tests/test_app_utils.py::test_get_significance_marker[0.2-ns]**
- **tests/test_app_utils.py::test_get_significance_marker[0.04-*]**
- **tests/test_app_utils.py::test_get_significance_marker[0.009-**]**
- **tests/test_app_utils.py::test_get_significance_marker[0.0009-***]**
- **tests/test_app_utils.py::test_calculate_correlation_basic_and_insufficient_data**
- **tests/test_app_utils.py::test_load_experiment_metrics_and_load_uncertainty_logs_and_merge_logs**
- **tests/test_app_utils.py::test_load_batch_logs_from_experiments_dir**
- **tests/test_app_utils.py::test_end_to_end_pipeline_on_sample_logs**
- **tests/test_model_utils_litellm.py::test_encode_image_to_base64_from_path_and_pil_and_array**
- **tests/test_model_utils_litellm.py::test_check_litellm_running**
- **tests/test_model_utils_litellm.py::test_check_litellm_not_running**
- **tests/test_model_utils_litellm.py::test_prepare_messages_builds_expected_structure**
- **tests/test_model_utils_litellm.py::test_litellm_request_builds_payload_and_parses_response**
- **tests/test_model_utils_litellm.py::test_litellm_request_gpt5_nano_temperature_override**
- **tests/test_model_utils_litellm.py::test_request_model_and_request_gpt_wrappers**
- **tests/test_prompt_library.py::test_system_prompt_library_loads_txt_files**
- **tests/test_prompt_library.py::test_system_prompt_library_load_and_prepare_prompt**
- **tests/test_prompt_library.py::test_system_prompt_library_prepare_prompt_missing_raises**
- **tests/test_prompt_library.py::test_system_prompt_library_prepare_variant_prompts**
- **tests/test_prompt_library.py::test_system_prompt_library_read_prompt_from_file**
- **tests/test_tracker.py::test_generate_experiment_id_is_deterministic_and_formatted**
- **tests/test_tracker.py::test_detect_experiment_group_baseline_vs_uncertainty**
- **tests/test_tracker.py::test_estimate_scenario_difficulty[1-1.0-False-easy]**
- **tests/test_tracker.py::test_estimate_scenario_difficulty[20-0.8-False-medium]**
- **tests/test_tracker.py::test_estimate_scenario_difficulty[15-0.5-True-hard]**
- **tests/test_tracker.py::test_experiment_tracker_model_category_and_summary**
- **tests/test_tracker.py::test_experiment_tracker_metadata_and_uncertainty_snapshot**
- **tests/test_tracker.py::test_experiment_tracker_save_uncertainty_log**
- **tests/test_tracker.py::test_grasp_stats_tracker_record_and_rates**
- **tests/test_tracker.py::test_grasp_stats_tracker_reset**
- **tests/test_uncertainty_analyzer.py::test_cluster_equivalent_responses_single_text**
- **tests/test_uncertainty_analyzer.py::test_cluster_equivalent_responses_multiple_texts**
- **tests/test_uncertainty_analyzer.py::test_calculate_posterior_with_logprobs**
- **tests/test_uncertainty_analyzer.py::test_calculate_posterior_without_logprobs**
- **tests/test_uncertainty_analyzer.py::test_compute_entropy_basic**
- **tests/test_uncertainty_analyzer.py::test_extract_final_answer_key_matches_pattern_and_fallback**
- **tests/test_uncertainty_analyzer.py::test_parse_confidence_from_response[confidence: 0.85-0.85]**
- **tests/test_uncertainty_analyzer.py::test_parse_confidence_from_response[Confidence score: 0.7.-0.7]**
- **tests/test_uncertainty_analyzer.py::test_parse_confidence_from_response[no confidence here-None]**
- **tests/test_uncertainty_analyzer.py::test_extract_uncertainty_descriptors[Uncertainty: "object is occluded by another object"-object is occluded by another object]**
- **tests/test_uncertainty_analyzer.py::test_extract_uncertainty_descriptors[Uncertainty: high occlusion\nNext line-high occlusion]**
- **tests/test_uncertainty_analyzer.py::test_extract_uncertainty_descriptors[No uncertainty section-None]**
- **tests/test_uncertainty_analyzer.py::test_extract_metadata_uses_defaults_and_parsed_values**

### Skipped (1)

- **tests/test_app_module.py::test_app_import_smoke**

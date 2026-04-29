import os
import json
from datetime import datetime

# ==============================================================================
# Configuration
# ==============================================================================

# ==> Unified list of experiments to process.
# Fill in the metadata for each experiment. `None` will become `null` in the JSON.

EXPERIMENTS = [
    {
        "path": "vlm2vec_exps/VLM2Vec-Qwen2VL-V2.0-2B/",
        "metadata": {
            "model_name": "VLM2Vec-Qwen2VL-V2.0-2B",
            "model_size": "2B parameters",
            "embedding_dimension": None, # Please fill in
            "max_length_tokens": None,   # Please fill in
            "model_release_date": "2025-04-01", # Please adjust this date
            "score_source": "",           # e.g., "Self-Reported" or "TIGER-Lab"
            "url": ""                    # e.g., Paper, GitHub, or Hugging Face link
        }
    },
    {
        "path": "vlm2vec_exps/VLM2Vec-Qwen2VL-V2.1-2B/",
        "metadata": {
            "model_name": "VLM2Vec-Qwen2VL-V2.1-2B",
            "model_size": "2B parameters",
            "embedding_dimension": None, # Please fill in
            "max_length_tokens": None,   # Please fill in
            "model_release_date": "2025-05-15", # Please adjust this date
            "score_source": "",           # e.g., "Self-Reported" or "TIGER-Lab"
            "url": ""                    # e.g., Paper, GitHub, or Hugging Face link
        }
    },
]

# ==============================================================================
# TODO: Your models' metadata goes here. Please fill in the required fields.
# ==============================================================================

EXPERIMENTS = [
    {
        "path": "",
        "metadata": {
            "model_name": "UME-R1-7B",
            "model_backbone": "Qwen2-VL-7B-Instruct",
            "model_size": "7B parameters",
            "embedding_dimension": 4096,
            "max_length_tokens": 8192,
            "model_release_date": "2025-07-17",
            "data_source": "Self-Reported",
            "url": ""
        }
    }
]

# ==============================================================================
# Main Processing Logic (No changes needed below this line)
# ==============================================================================


# Define the datasets grouped by modality
modality2dataset = {
    "image": [
        "ImageNet-1K", "N24News", "HatefulMemes", "VOC2007", "SUN397", "Place365", "ImageNet-A", "ImageNet-R", "ObjectNet", "Country211",
        "OK-VQA", "A-OKVQA", "DocVQA", "InfographicsVQA", "ChartQA", "Visual7W", "ScienceQA", "VizWiz", "GQA", "TextVQA",
        "VisDial", "CIRR", "VisualNews_t2i", "VisualNews_i2t", "MSCOCO_t2i", "MSCOCO_i2t", "NIGHTS", "WebQA", "FashionIQ", "Wiki-SS-NQ", "OVEN", "EDIS",
        "MSCOCO", "RefCOCO", "RefCOCO-Matching", "Visual7W-Pointing"],
    "video": [
        "K700", "SmthSmthV2", "HMDB51", "UCF101", "Breakfast",
        "MVBench", "Video-MME", "NExTQA", "EgoSchema", "ActivityNetQA",
        "DiDeMo", "MSR-VTT", "MSVD", "VATEX", "YouCook2",
        "QVHighlight", "Charades-STA", "MomentSeeker",
    ],
    "visdoc": [
        "ViDoRe_arxivqa", "ViDoRe_docvqa", "ViDoRe_infovqa", "ViDoRe_tabfquad", "ViDoRe_tatdqa", "ViDoRe_shiftproject",
        "ViDoRe_syntheticDocQA_artificial_intelligence", "ViDoRe_syntheticDocQA_energy", "ViDoRe_syntheticDocQA_government_reports", "ViDoRe_syntheticDocQA_healthcare_industry",
        "ViDoRe_esg_reports_human_labeled_v2", "ViDoRe_biomedical_lectures_v2_multilingual", "ViDoRe_economics_reports_v2_multilingual", "ViDoRe_esg_reports_v2_multilingual",
        "VisRAG_ArxivQA", "VisRAG_ChartQA", "VisRAG_MP-DocVQA", "VisRAG_SlideVQA", "VisRAG_InfoVQA", "VisRAG_PlotQA",
        "ViDoSeek-page", "ViDoSeek-doc", "MMLongBench-page", "MMLongBench-doc"
    ]
}
# in_domain_results = {
#     "ImageNet-1K": read_json_score(os.path.join(dir, "ImageNet-1K_score.json")),
#     "HatefulMemes": read_json_score(os.path.join(dir, "HatefulMemes_score.json")),
#     "SUN397": read_json_score(os.path.join(dir, "SUN397_score.json")),
#     "N24News": read_json_score(os.path.join(dir, "N24News_score.json")),
#     "VOC2007": read_json_score(os.path.join(dir, "VOC2007_score.json")),
#     "OK-VQA": read_json_score(os.path.join(dir, "OK-VQA_score.json")),
#     "A-OKVQA": read_json_score(os.path.join(dir, "A-OKVQA_score.json")),
#     "DocVQA": read_json_score(os.path.join(dir, "DocVQA_score.json")),
#     "InfographicsVQA": read_json_score(os.path.join(dir, "InfographicsVQA_score.json")),
#     "ChartQA": read_json_score(os.path.join(dir, "ChartQA_score.json")),
#     "Visual7W": read_json_score(os.path.join(dir, "Visual7W_score.json")),
#     "VisDial": read_json_score(os.path.join(dir, "VisDial_score.json")),
#     "CIRR": read_json_score(os.path.join(dir, "CIRR_score.json")),
#     "NIGHTS": read_json_score(os.path.join(dir, "NIGHTS_score.json")),
#     "WebQA": read_json_score(os.path.join(dir, "WebQA_score.json")),
#     "VisualNews_i2t": read_json_score(os.path.join(dir, "VisualNews_i2t_score.json")),
#     "VisualNews_t2i": read_json_score(os.path.join(dir, "VisualNews_t2i_score.json")),
#     "MSCOCO_t2i": read_json_score(os.path.join(dir, "MSCOCO_t2i_score.json")),
#     "MSCOCO_i2t": read_json_score(os.path.join(dir, "MSCOCO_i2t_score.json")),
#     "MSCOCO": read_json_score(os.path.join(dir, "MSCOCO_score.json")),
# }
# out_domain_results = {
#     "Place365": read_json_score(os.path.join(dir, "Place365_score.json")),
#     "ImageNet-A": read_json_score(os.path.join(dir, "ImageNet-A_score.json")),
#     "ImageNet-R": read_json_score(os.path.join(dir, "ImageNet-R_score.json")),
#     "ObjectNet": read_json_score(os.path.join(dir, "ObjectNet_score.json")),
#     "Country211": read_json_score(os.path.join(dir, "Country211_score.json")),
#     "ScienceQA": read_json_score(os.path.join(dir, "ScienceQA_score.json")),
#     "GQA": read_json_score(os.path.join(dir, "GQA_score.json")),
#     "TextVQA": read_json_score(os.path.join(dir, "TextVQA_score.json")),
#     "VizWiz": read_json_score(os.path.join(dir, "VizWiz_score.json")),
#     "FashionIQ": read_json_score(os.path.join(dir, "FashionIQ_score.json")),
#     "Wiki-SS-NQ": read_json_score(os.path.join(dir, "Wiki-SS-NQ_score.json")),
#     "OVEN": read_json_score(os.path.join(dir, "OVEN_score.json")),
#     "EDIS": read_json_score(os.path.join(dir, "EDIS_score.json")),
#     "RefCOCO": read_json_score(os.path.join(dir, "RefCOCO_score.json")),
#     "Visual7W-Pointing": read_json_score(os.path.join(dir, "Visual7W-Pointing_score.json")),
#     "RefCOCO-Matching": read_json_score(os.path.join(dir, "RefCOCO-Matching_score.json")),
# }
type2dataset = {
    "Image-CLS": [
        "ImageNet-1K", "N24News", "HatefulMemes", "VOC2007", "SUN397", "Place365", "ImageNet-A", "ImageNet-R", "ObjectNet", "Country211"
    ],
    "Image-QA": [
        "OK-VQA", "A-OKVQA", "DocVQA", "InfographicsVQA", "ChartQA", "Visual7W", "ScienceQA", "VizWiz", "GQA", "TextVQA"
    ],
    "Image-RET": [
        "VisDial", "CIRR", "VisualNews_t2i", "VisualNews_i2t", "MSCOCO_t2i", "MSCOCO_i2t", "NIGHTS", "WebQA", "FashionIQ", "Wiki-SS-NQ", "OVEN", "EDIS"
    ],
    "Image-GD": [
        "MSCOCO", "RefCOCO", "RefCOCO-Matching", "Visual7W-Pointing"
    ],
    "Image-InD": [
        "ImageNet-1K", "HatefulMemes", "SUN397", "N24News", "VOC2007",
        "OK-VQA", "A-OKVQA", "DocVQA", "InfographicsVQA", "ChartQA", "Visual7W", "VisDial", "CIRR", "NIGHTS", "WebQA", "VisualNews_i2t", "VisualNews_t2i", "MSCOCO_i2t", "MSCOCO_t2i",
        "MSCOCO",
    ],
    "Image-OutD": [
        "Place365", "ImageNet-A", "ImageNet-R", "ObjectNet", "Country211",
        "ScienceQA", "GQA", "TextVQA", "VizWiz", "FashionIQ", "Wiki-SS-NQ", "OVEN", "EDIS",
        "RefCOCO", "RefCOCO-Matching", "Visual7W-Pointing"
    ],
    "Video-CLS": [
        "K700", "SmthSmthV2", "HMDB51", "UCF101", "Breakfast"
    ],
    "Video-QA": [
        "MVBench","Video-MME", "NExTQA", "EgoSchema", "ActivityNetQA",
    ],
    "Video-RET": [
        "DiDeMo", "MSR-VTT", "MSVD", "VATEX", "YouCook2",
    ],
    "Video-MRET": [
        "QVHighlight", "Charades-STA", "MomentSeeker"
    ],
    "VisDoc-VDRv1": [
        "ViDoRe_arxivqa", "ViDoRe_docvqa", "ViDoRe_infovqa", "ViDoRe_tabfquad", "ViDoRe_tatdqa", "ViDoRe_shiftproject",
        "ViDoRe_syntheticDocQA_artificial_intelligence", "ViDoRe_syntheticDocQA_energy", "ViDoRe_syntheticDocQA_government_reports", "ViDoRe_syntheticDocQA_healthcare_industry",
    ],
    "VisDoc-VDRv2": [
        "ViDoRe_esg_reports_human_labeled_v2", "ViDoRe_biomedical_lectures_v2_multilingual", "ViDoRe_economics_reports_v2_multilingual", "ViDoRe_esg_reports_v2_multilingual",
    ],
    "VisDoc-VR": [
        "VisRAG_ArxivQA", "VisRAG_ChartQA", "VisRAG_MP-DocVQA", "VisRAG_SlideVQA", "VisRAG_InfoVQA", "VisRAG_PlotQA",
    ],
    "VisDoc-OOD": [
        "ViDoSeek-page", "ViDoSeek-doc", "MMLongBench-page", "MMLongBench-doc"
    ],
}

modality2metric = {
    "image": "hit@1",
    "video": "hit@1",
    "visdoc": "ndcg_linear@5",
}
modalities = ["image", "video", "visdoc"] # Process in this order

qry_mode = "disc"
tgt_mode = "disc"

qry_mode = "gen"
tgt_mode = "gen"
# modalities = ["image"] # Process in this order

for experiment in EXPERIMENTS:
    base_path = experiment['path']
    experiment_metadata = experiment['metadata']
    experiment_name_for_log = os.path.basename(base_path.strip('/'))

    current_experiment_scores = {}

    print(f"\nProcessing experiment: {experiment_name_for_log}")
    print(f"Path: {base_path}")

    for modality in modalities:
        current_experiment_scores[modality] = {}
        modality_specific_result_dir = os.path.join(base_path, modality)

        for dataset_name in modality2dataset.get(modality, []):
            current_experiment_scores[modality][dataset_name] = "FILE_N/A" # Initialize

        if not os.path.isdir(modality_specific_result_dir):
            print(f"    Directory not found: {modality_specific_result_dir}")
            for dataset_name in modality2dataset.get(modality, []):
                current_experiment_scores[modality][dataset_name] = "DIR_N/A"
            continue

        for filename in os.listdir(modality_specific_result_dir):
            if filename.endswith(f"{qry_mode}_{tgt_mode}_score.json"):
                score_file_path = os.path.join(modality_specific_result_dir, filename)
                dataset_name_from_file = None
                for known_dataset in modality2dataset.get(modality, []):
                    if filename == f"{known_dataset}_{qry_mode}_{tgt_mode}_score.json":
                        dataset_name_from_file = known_dataset
                        break

                if dataset_name_from_file:
                    try:
                        with open(score_file_path, "r") as f:
                            score_data = json.load(f)
                            current_experiment_scores[modality][dataset_name_from_file] = score_data
                    except json.JSONDecodeError:
                        print(f"      Error decoding JSON from {score_file_path}")
                        current_experiment_scores[modality][dataset_name_from_file] = "JSON_ERROR"
                    except Exception as e:
                        print(f"      Error reading file {score_file_path}: {e}")
                        current_experiment_scores[modality][dataset_name_from_file] = "READ_ERROR"

    # --- Construct and Save the Final JSON Report ---
    final_metadata = experiment_metadata.copy()
    final_metadata['report_generated_date'] = datetime.now().isoformat()
    final_output = {
        "metadata": final_metadata,
        "metrics": current_experiment_scores
    }

    output_json_path = os.path.join(base_path, f"{final_metadata['model_name']}.json")
    try:
        with open(output_json_path, "w") as f:
            json.dump(final_output, f, indent=4)
        print(f"  Report for '{experiment_name_for_log}' saved to: {output_json_path}")
    except Exception as e:
        print(f"  Error saving JSON report for '{experiment_name_for_log}' to {output_json_path}: {e}")


    # --- Print detailed main scores per dataset for easy copy to spreadsheet ---
    print(f"\n  --- Detailed Main Scores for Spreadsheet (Experiment: {experiment_name_for_log}) ---")
    for modality in modalities:
        main_metric_key = modality2metric[modality]
        for dataset_name in modality2dataset.get(modality, []):
            score_to_print_val = "NOT_FOUND_IN_RESULTS"
            modality_data = current_experiment_scores.get(modality, {})
            score_info = modality_data.get(dataset_name)

            if isinstance(score_info, dict):
                metric_value = score_info.get(main_metric_key)
                if isinstance(metric_value, (int, float)):
                    score_to_print_val = f"{metric_value:.4f}"
                else:
                    score_to_print_val = f"METRIC_KEY_MISSING ({main_metric_key})"
            elif isinstance(score_info, str):
                score_to_print_val = score_info

            print(f"{dataset_name}\t{score_to_print_val}")
        print("")

    # --- Print average scores and missing datasets per modality ---
    print(f"\n  --- Summary for Experiment: {experiment_name_for_log} ---")
    for modality in modalities:
        if modality not in current_experiment_scores:
            print(f"    Modality '{modality.upper()}' not processed.")
            continue
        main_metric_key = modality2metric[modality]
        modality_data = current_experiment_scores[modality]
        collected_metric_values = []
        datasets_missing_score_file = []
        datasets_file_found_metric_missing = []

        for dataset_name in modality2dataset.get(modality, []):
            score_info = modality_data.get(dataset_name)
            if isinstance(score_info, dict):
                metric_value = score_info.get(main_metric_key)
                if isinstance(metric_value, (int, float)):
                    if modality == "visdoc":
                        collected_metric_values.append(metric_value-0.000)
                    else:
                        collected_metric_values.append(metric_value)
                else:
                    datasets_file_found_metric_missing.append(f"{dataset_name} (metric '{main_metric_key}' missing/invalid)")
            else:
                datasets_missing_score_file.append(f"{dataset_name} (status: {score_info if score_info else 'Not Processed'})")

        if collected_metric_values:
            average_score = sum(collected_metric_values) / len(collected_metric_values)
            print(f"      Average of {modality.upper()}\t- {main_metric_key}:\t{average_score:.4f} (from {len(collected_metric_values)} datasets)")
        else:
            print(f"      Average of {modality.upper()}\t-  {main_metric_key}:\tN/A (no valid scores found)")

        if datasets_missing_score_file:
            print(f"      Datasets with missing/errored score files:")
            for ds_status in datasets_missing_score_file: print(f"        - {ds_status}")
        if datasets_file_found_metric_missing:
            print(f"      Score files found but main metric ('{main_metric_key}') missing/invalid:")
            for ds_status in datasets_file_found_metric_missing: print(f"        - {ds_status}")
    # --- Print average scores for each type of dataset ---
    
    print(f"\n  --- Average Scores by Dataset Type for Experiment: {experiment_name_for_log} ---")
    for dataset_type, datasets in type2dataset.items():
        collected_scores = []
        for dataset_name in datasets:
            for modality in modalities:
                modality_data = current_experiment_scores.get(modality, {})
                score_info = modality_data.get(dataset_name)
                if isinstance(score_info, dict):
                    metric_value = score_info.get(modality2metric[modality])
                    if isinstance(metric_value, (int, float)):
                        if modality == "visdoc":
                            collected_scores.append(metric_value-0.000)
                        else:
                            collected_scores.append(metric_value)

        if collected_scores:
            average_score = sum(collected_scores) / len(collected_scores)
            print(f"    Average of {dataset_type}:\t{average_score:.4f} (from {len(collected_scores)} datasets)")
        else:
            print(f"    Average of {dataset_type}:\tN/A (no valid scores found)")



    # --- Calculate and print the overall average score across all modalities ---
    overall_collected_scores = []
    for modality in modalities:
        main_metric_key = modality2metric[modality]
        modality_data = current_experiment_scores.get(modality, {})
        for dataset_name in modality2dataset.get(modality, []):
            score_info = modality_data.get(dataset_name)
            if isinstance(score_info, dict):
                metric_value = score_info.get(main_metric_key)
                if isinstance(metric_value, (int, float)):
                    if modality == "visdoc":
                        overall_collected_scores.append(metric_value-0.000)
                    else:
                        overall_collected_scores.append(metric_value)
    if overall_collected_scores:
        overall_average_score = sum(overall_collected_scores) / len(overall_collected_scores)
        print(f"      Overall Average Score across all modalities: {overall_average_score:.4f} (from {len(overall_collected_scores)} datasets)")
    else:
        print(f"      Overall Average Score across all modalities: N/A (no valid scores found)")

    print("\n" * 2)  # Add some space between experiments


print("\nProcessing complete.")
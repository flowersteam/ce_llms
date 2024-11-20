import datasets
import glob

for g_i in range(0, 20, 5):
    d = datasets.load_from_disk(glob.glob(
        f"dev_results/human_ai_ratio_v3_acc_1*participants_1*/generated_250_*/seed_0*/gen_{g_i}/part_0/full_output_dataset"
    )[0]
    )
    print(f"GEN: {g_i}")
    d_sample = d.select(range(5))
    for i, (ins, t) in enumerate(zip(d_sample['instruction'], d_sample['text'])):
        print(f"\tInstr:{ins}\n\tResp:{t}\n")

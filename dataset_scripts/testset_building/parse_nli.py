import json
import argparse


def main(args):
    ret_data = []
    invalid = 0
    with open(args.infile) as f:
        data = json.load(f)
        for d in data:
            text = d["synthesized_case"]
            premises, hypothesis, label = tuple(text.split("\n"))
            ret_data.append(
                {
                    "premises": premises.replace("premises: ",""),
                    "hypothesis": hypothesis.replace("hypothesis: ",""),
                    "label": label.replace("label: ","")
                }
            )


    with open(args.outfile,"w") as fout:
        for d in ret_data:
            fout.write(json.dumps(d,ensure_ascii=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--infile")
    parser.add_argument("--outfile")

    args = parser.parse_args()
    main(args)                
        


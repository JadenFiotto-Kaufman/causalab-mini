"""A roles-and-interventions document, rewritten in the steps-first format.

Test-only, and only for as long as the old format exists: it is the
instrument `test_parity.py` measures the new front end with. It works on
the raw JSON, so a `{"sweep": …}` wrapper rides along wherever it sits and
every point keeps its label.

The rules, one per construct of the old format:

- every `(model, input)` forward of a step's intervention is a forward step,
  in the order the old compiler scheduled them — named after the old step
  when it is the only one, else after its model (its role, for `original`);
- a model's writes are an intervention written in place on its forward;
  a read is the forward's own, and an operand is `<forward>.<read>`;
- `decode: N` makes every forward of the intervention a generate step that
  holds EOS off, which is what the engines did by hand;
- each metric is a metric step, its columns the base role's dataset's;
- an output reduced to a mean or a basis is a reduce step of its name; one
  kept as it is, is its read's reference;
- a fit's body is its intervention's forwards and metrics, `rows` and
  `eval.rows` are its data and `eval.data`, and a featurizer it trains is
  `<fit>.<name>` everywhere;
- a save moves to `steps.saves`, and a list step's saves keep their
  directory (`compare/patching/…`) in their file path.

A name already taken is prefixed with its old step's (or list entry's) name.
"""

from __future__ import annotations

import copy
from typing import Any

from causalab_mini.plan import sweep
from causalab_mini.plan.build import _Experiment, _schedule
from causalab_mini.plan.spec_v2 import Spec as OldSpec

Json = dict[str, Any]


def convert(raw: Json) -> Json:
    old = OldSpec.model_validate(sweep.points(raw)[0][1])  # a point: the schedule does not sweep
    trainers = {one: name for name, step in raw["steps"].items() if step["kind"] == "fit" for one in step["params"]}
    data: dict[str, str] = {}  # ref -> dataset name

    def dataset(ref: str) -> str:
        if ref not in data:
            name = ref.partition("#")[2] or ref.rpartition("/")[2]
            while name in data.values():
                name += "_"
            data[ref] = name
        return data[ref]

    def featurized(op: Json) -> Json:
        one = {key: value for key, value in op.items() if key not in ("model", "input")}
        if one.get("featurizer") in trainers:
            one["featurizer"] = f"{trainers[one['featurizer']]}.{one['featurizer']}"
        return one

    steps: Json = {}
    saves: dict[str, str] = {}
    aliases: dict[str, str] = {}  # an old output's name -> its reference

    def experiment(old_name: str, label: str, one: Json, rows: dict[str, str], scope: Json, alone: bool) -> dict[str, str]:
        """Write one intervention's forwards and metrics into `scope`, and
        return what each old name became: `<read>`, `<metric>` and
        `<model>.generated`."""
        units = _schedule(_Experiment.of_spec(old, old.interventions[label]))
        forwards: dict[tuple[str, str], str] = {}
        for model, role in units:
            wanted = old_name if (alone and len(units) == 1) else (role if model == "original" else model)
            forwards[(model, role)] = _free(wanted, scope, old_name if alone else label)
        home = {name: forwards[(read.get("model", "original"), read["input"])] for name, read in one["reads"].items()}
        for (model, role), name in forwards.items():
            step: Json = {
                "kind": "generate" if one.get("decode") else "forward",
                "data": dataset(rows[role]),
                "field": raw["roles"][role]["field"],
            }
            if one.get("decode"):
                step.update(max_new_tokens=one["decode"], min_new_tokens=one["decode"], do_sample=False)
            writes = {}
            for write_name in one.get("models", {}).get(model, {}).get("writes", []):
                write = featurized(copy.deepcopy(one["writes"][write_name]))
                operand = write.get("operand")
                if isinstance(operand, str):
                    write["operand"] = f"{home[operand]}.{operand}"
                elif isinstance(operand, dict):
                    write["operand"] = aliases[operand["ref"]]
                writes[write_name] = write
            if writes:
                step["interventions"] = {"writes": writes}
            step["reads"] = {
                read_name: featurized(copy.deepcopy(read))
                for read_name, read in one["reads"].items()
                if home[read_name] == name
            }
            scope[name] = step
        became = {name: f"{forward}.{name}" for name, forward in home.items()}
        became |= {f"{model}.generated": name for (model, _), name in forwards.items()}
        for metric_name, metric in one.get("metrics", {}).items():
            columns = {key: f"{dataset(rows['base'])}.{value}" for key, value in metric.items() if key not in ("kind", "of", "token_form")}
            step = {"kind": "metric", "metric": metric["kind"], "of": became[metric["of"]], **columns}
            if "token_form" in metric:
                step["token_form"] = metric["token_form"]
            became[metric_name] = _free(metric_name, scope, old_name if alone else label)
            scope[became[metric_name]] = step
        return became

    for old_name, old_step in raw["steps"].items():
        if old_step["kind"] == "weights":
            for save in old_step.get("saves", []):
                saves[f"{trainers[save['value']]}.{save['value']}"] = save["file_path"]
            continue
        given = old_step["interventions"]
        listed = isinstance(given, list)
        for index, entry in enumerate(given if listed else [given]):
            label = entry if isinstance(entry, str) else (f"{old_name}[{index}]" if listed else old_name)
            one = raw["interventions"][entry] if isinstance(entry, str) else entry
            if old_step["kind"] == "fit":
                body: Json = {}
                became = experiment(old_name, label, one, old_step["rows"], body, alone=False)
                rows = old_step["rows"]
                steps[old_name] = {
                    "kind": "fit",
                    "train": old_step["params"],
                    "objective": [[weight, became.get(term, term)] for weight, term in old_step["objective"]],
                    **({"anneal": old_step["anneal"]} if "anneal" in old_step else {}),
                    "epochs": old_step["epochs"],
                    "batch_size": old_step["pairs"],
                    "seed": old_step.get("seed", 0),
                    "optimizer": old_step["optimizer"],
                    "early_stop": {**old_step["early_stop"], "metric": became[old_step["early_stop"]["metric"]]},
                    "eval": {"data": {dataset(rows[role]): dataset(ref) for role, ref in old_step["eval"]["rows"].items()}},
                    "steps": body,
                }
                for save in old_step["eval"].get("saves", []):
                    saves[f"{old_name}.{became[save['value']]}"] = save["file_path"]
                continue
            became = experiment(old_name, label, one, old_step["rows"], steps, alone=not listed)
            for output_name, output in old_step.get("outputs", {}).items():
                output = {"read": output} if isinstance(output, str) else output
                if output.get("reduce", "none") == "none":
                    became[output_name] = aliases[output_name] = became[output["read"]]
                    continue
                name = _free(output_name, steps, old_name)
                steps[name] = {"kind": "reduce", "reduce": output["reduce"], "of": became[output["read"]], **({"k": output["k"]} if "k" in output else {})}
                became[output_name] = aliases[output_name] = name
            for save in old_step.get("saves", []):
                # a list step's save names its entry, or is the one entry's that has it
                owner, _, value = save["value"].rpartition("/")
                child = entry if isinstance(entry, str) else label
                if value in became and owner in ("", child):
                    saves[became[value]] = f"{old_name}/{child}/{save['file_path']}" if listed else save["file_path"]
    steps["saves"] = saves
    converted: Json = {"header": raw.get("header", {}), "model": raw["model"]}
    converted["data"] = {name: {"path": ref} for ref, name in data.items()}
    converted["sites"] = raw["sites"]
    if raw.get("featurizers"):
        converted["featurizers"] = raw["featurizers"]
    converted["steps"] = steps
    return converted


def _free(name: str, scope: Json, prefix: str) -> str:
    """`name`, or — when the scope already has it — `<prefix>_<name>`."""
    while name in scope:
        name = f"{prefix}_{name}"
    return name

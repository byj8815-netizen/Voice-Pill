import json
from pathlib import Path


def _read_json(path):
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)


class PillLabelMap:
    def __init__(self, label_path, metadata_root=None):
        label_data = _read_json(label_path)
        rows = label_data.get("pill_label_path_sharp_score")
        if not isinstance(rows, list):
            raise ValueError("Label JSON must contain a 'pill_label_path_sharp_score' list.")

        self.class_to_pill = {}
        for row in rows:
            if not isinstance(row, list) or len(row) < 2:
                raise ValueError(f"Invalid pill label row: {row!r}")
            class_id, pill_id = int(row[0]), str(row[1])
            if class_id in self.class_to_pill:
                raise ValueError(f"Duplicate class ID in label JSON: {class_id}")
            self.class_to_pill[class_id] = pill_id
        self.pill_to_class = {pill_id: class_id for class_id, pill_id in self.class_to_pill.items()}

        self.pill_to_name = {}
        if metadata_root is not None:
            self.pill_to_name = self._load_names(Path(metadata_root))

    def _load_names(self, metadata_root):
        names = {}
        if not metadata_root.is_dir():
            return names

        for pill_id in set(self.class_to_pill.values()):
            pill_dir = metadata_root / pill_id
            if not pill_dir.is_dir():
                continue
            for metadata_path in pill_dir.glob("*.json"):
                try:
                    metadata = _read_json(metadata_path)
                except (OSError, json.JSONDecodeError):
                    continue
                for image_info in metadata.get("images", []):
                    metadata_id = image_info.get("drug_N") or image_info.get("dl_mapping_code")
                    name = image_info.get("dl_name")
                    if metadata_id == pill_id and isinstance(name, str) and name.strip():
                        names[pill_id] = name.strip()
                        break
                if pill_id in names:
                    break
        return names

    def resolve(self, class_id):
        pill_id = self.class_to_pill.get(int(class_id))
        return {
            "pill_id": pill_id,
            "pill_name": self.pill_to_name.get(pill_id) if pill_id is not None else None,
            "pill_name_status": "mapped" if pill_id in self.pill_to_name else "mapping_missing",
        }
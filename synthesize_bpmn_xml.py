import os
import json
import uuid
import copy
import xml.etree.ElementTree as ET

# =================配置区域=================
POOL_DIR = "./auto_pool"
COMPLEX_DATA_DIR = "./synthetic_complex_dataset"
OUTPUT_XML_DIR = "./synthetic_complex_dataset/bpmn"
# ========================================

NAMESPACES = {
    'bpmn': 'http://www.omg.org/spec/BPMN/20100524/MODEL',
    'bpmndi': 'http://www.omg.org/spec/BPMN/20100524/DI',
    'omgdc': 'http://www.omg.org/spec/DD/20100524/DC',
    'omgdi': 'http://www.omg.org/spec/DD/20100524/DI',
    'xsi': 'http://www.w3.org/2001/XMLSchema-instance'
}
for prefix, uri in NAMESPACES.items():
    ET.register_namespace(prefix, uri)


class BpmnMerger:
    def __init__(self):
        self.atom_map = self._load_atom_map()

    def _load_atom_map(self):
        mapping = {}
        if not os.path.exists(POOL_DIR): return {}
        for f in os.listdir(POOL_DIR):
            if f.endswith('.json'):
                try:
                    with open(os.path.join(POOL_DIR, f), 'r', encoding='utf-8') as file:
                        data = json.load(file)
                        xml_content = data.get("bpmn_xml") or data.get("xml")
                        if xml_content: mapping[data["id"]] = xml_content
                except:
                    pass
        return mapping

    def _parse_xml(self, xml_str):
        return ET.fromstring(xml_str)

    def _tag(self, elem):
        return elem.tag.split('}')[-1] if '}' in elem.tag else elem.tag

    def _smart_process_xml(self, root, new_prefix):
        """
        【智能修复逻辑】
        1. 扫描现有 ID (例如: "abc_Task_1")
        2. 猜测原始 ID (例如: "Task_1")
        3. 建立映射: "Task_1" -> "new_prefix_Task_1"
        4. 同时映射: "abc_Task_1" -> "new_prefix_Task_1"
        """
        id_map = {}

        # --- 阶段 1: 智能构建映射表 ---
        for elem in root.iter():
            if 'id' in elem.attrib:
                current_id = elem.attrib['id']

                # 生成最终的新 ID
                # 为了避免 ID 无限变长，我们只保留原始部分 + 新前缀
                # 假设 current_id 是 "oldprefix_OriginalID"
                if '_' in current_id:
                    # 尝试去掉第一个下划线前的部分作为 OriginalID
                    parts = current_id.split('_', 1)
                    original_id = parts[1]
                else:
                    original_id = current_id

                final_id = f"{new_prefix}{original_id}"

                # 关键：同时映射“当前ID”和“猜测的原始ID”
                id_map[current_id] = final_id
                id_map[original_id] = final_id

        print(f"    - 构建了 {len(id_map)} 条 ID 映射规则")

        # --- 阶段 2: 暴力替换 ---
        replace_count = 0
        for elem in root.iter():
            # 1. 替换属性
            for key, val in list(elem.attrib.items()):
                # 如果值在映射表中，直接替换
                if val in id_map:
                    elem.attrib[key] = id_map[val]
                    replace_count += 1
                # 特殊处理：有些引用可能有命名空间前缀 (如 ns1:Task_1)
                elif ":" in val:
                    suffix = val.split(":")[-1]
                    if suffix in id_map:
                        elem.attrib[key] = id_map[suffix]
                        replace_count += 1

            # 2. 替换文本 (incoming/outgoing)
            if elem.text:
                text = elem.text.strip()
                if text in id_map:
                    elem.text = id_map[text]
                    replace_count += 1

        print(f"    - 成功修复并替换了 {replace_count} 处引用")
        return id_map

    def _get_bounds(self, root):
        min_x, min_y = float('inf'), float('inf')
        max_x, max_y = float('-inf'), float('-inf')
        found = False
        for elem in root.iter():
            tag = self._tag(elem)
            if tag in ['Bounds', 'waypoint']:
                try:
                    x, y = float(elem.get('x')), float(elem.get('y'))
                    if tag == 'Bounds':
                        w, h = float(elem.get('width')), float(elem.get('height'))
                        max_x, max_y = max(max_x, x + w), max(max_y, y + h)
                    else:
                        max_x, max_y = max(max_x, x), max(max_y, y)
                    min_x, min_y = min(min_x, x), min(min_y, y)
                    found = True
                except:
                    pass
        if not found: return 0, 0, 200, 100
        return min_x, min_y, max_x, max_y

    def _shift_diagram(self, root, offset_x, offset_y):
        for elem in root.iter():
            tag = self._tag(elem)
            if tag in ['Bounds', 'waypoint']:
                try:
                    elem.set('x', str(float(elem.get('x')) + offset_x))
                    elem.set('y', str(float(elem.get('y')) + offset_y))
                except:
                    pass

    def _find_anchors(self, process_node):
        start_id, end_id = None, None
        first_node, last_node = None, None

        for child in process_node:
            tag = self._tag(child)
            if tag == 'startEvent':
                start_id = child.get('id')
            elif tag == 'endEvent':
                end_id = child.get('id')

        for child in process_node:
            tag = self._tag(child)
            if tag == 'sequenceFlow':
                s, t = child.get('sourceRef'), child.get('targetRef')
                if start_id and s == start_id: first_node = t
                if end_id and t == end_id: last_node = s

        return start_id, end_id, first_node, last_node

    def _get_node_center(self, root, node_id):
        for elem in root.iter():
            if self._tag(elem) == 'BPMNShape':
                if elem.get('bpmnElement') == node_id:
                    for child in elem:
                        if self._tag(child) == 'Bounds':
                            x, y = float(child.get('x')), float(child.get('y'))
                            w, h = float(child.get('width')), float(child.get('height'))
                            return x, y, w, h
        return None

    def merge(self, complex_json_path):
        try:
            with open(complex_json_path, 'r', encoding='utf-8') as f:
                complex_data = json.load(f)
        except:
            return

        atom_ids = complex_data.get("atomic_components", [])
        bridge_logics = complex_data.get("bridge_logics", [])

        if not atom_ids: return

        print(f"\n=== 正在构建: {complex_data.get('id')} ===")

        base_xml = """<?xml version="1.0" encoding="UTF-8"?>
<definitions xmlns="http://www.omg.org/spec/BPMN/20100524/MODEL" xmlns:bpmndi="http://www.omg.org/spec/BPMN/20100524/DI" xmlns:omgdc="http://www.omg.org/spec/DD/20100524/DC" xmlns:omgdi="http://www.omg.org/spec/DD/20100524/DI" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" targetNamespace="http://www.signavio.com" xsi:schemaLocation="http://www.omg.org/spec/BPMN/20100524/MODEL http://www.omg.org/spec/BPMN/2.0/20100501/BPMN20.xsd">
   <process id="Process_Merged" isExecutable="false"></process>
   <bpmndi:BPMNDiagram id="BPMNDiagram_1">
      <bpmndi:BPMNPlane id="BPMNPlane_1" bpmnElement="Process_Merged"></bpmndi:BPMNPlane>
   </bpmndi:BPMNDiagram>
</definitions>"""

        merged_root = ET.fromstring(base_xml)
        process_node = None
        plane_node = None
        for elem in merged_root.iter():
            if self._tag(elem) == 'process': process_node = elem
            if self._tag(elem) == 'BPMNPlane': plane_node = elem

        global_cursor_x = 50.0
        prev_last_node_info = None

        for i, atom_id in enumerate(atom_ids):
            xml_str = self.atom_map.get(atom_id)
            if not xml_str: continue
            try:
                atom_root = self._parse_xml(xml_str)
            except:
                continue

            print(f"  处理原子 [{i}]: {atom_id}")

            # 1. 智能修复 ID
            self._smart_process_xml(atom_root, f"atom_{i}_")

            # 2. 归一化
            min_x, min_y, max_x, max_y = self._get_bounds(atom_root)
            self._shift_diagram(atom_root, -min_x, -min_y)

            # 3. 平移
            height = max_y - min_y
            y_offset = 300 - (height / 2)
            self._shift_diagram(atom_root, global_cursor_x, y_offset)

            # 4. 查找 Process
            atom_process = None
            for elem in atom_root.iter():
                if self._tag(elem) == 'process':
                    atom_process = elem
                    break
            if atom_process is None: continue

            # 5. 查找锚点
            start_evt, end_evt, first_node, last_node = self._find_anchors(atom_process)

            # 6. 迁移元素
            is_first = (i == 0)
            is_last = (i == len(atom_ids) - 1)

            for child in list(atom_process):
                tag = self._tag(child)
                if tag == 'startEvent' and not is_first: continue
                if tag == 'endEvent' and not is_last: continue
                if tag == 'sequenceFlow':
                    s, t = child.get('sourceRef'), child.get('targetRef')
                    if not is_first and s == start_evt: continue
                    if not is_last and t == end_evt: continue
                process_node.append(copy.deepcopy(child))

            # 7. 迁移 DI
            atom_plane = None
            for elem in atom_root.iter():
                if self._tag(elem) == 'BPMNPlane':
                    atom_plane = elem
                    break

            if atom_plane is not None:
                for shape in list(atom_plane):
                    eid = shape.get("bpmnElement")
                    if not is_first and eid == start_evt: continue
                    if not is_last and eid == end_evt: continue
                    plane_node.append(copy.deepcopy(shape))

            # 8. 桥接
            if not is_first and prev_last_node_info and first_node:
                prev_id, prev_x, prev_y = prev_last_node_info

                curr_geo = self._get_node_center(atom_root, first_node)
                if curr_geo:
                    curr_x, curr_y, curr_w, curr_h = curr_geo
                    curr_left_x = curr_x
                    curr_center_y = curr_y + curr_h / 2

                    bridge_text = "Transition"
                    if i - 1 < len(bridge_logics):
                        bridge_text = bridge_logics[i - 1]  # 直接使用完整文本
                        # full_text = bridge_logics[i - 1]
                        # bridge_text = (full_text[:30] + '...') if len(full_text) > 30 else full_text

                    bridge_id = f"Bridge_{i}_{uuid.uuid4().hex[:4]}"
                    task = ET.Element("bpmn:task", {"id": bridge_id, "name": bridge_text})
                    process_node.append(task)

                    mid_x = (prev_x + curr_left_x) / 2 - 50
                    mid_y = (prev_y + curr_center_y) / 2 - 40

                    shape = ET.Element("bpmndi:BPMNShape", {"id": f"{bridge_id}_di", "bpmnElement": bridge_id})
                    bounds = ET.Element("omgdc:Bounds",
                                        {"x": str(mid_x), "y": str(mid_y), "width": "100", "height": "80"})
                    shape.append(bounds)
                    plane_node.append(shape)

                    # 连线 1
                    f1_id = f"Flow_P2B_{i}"
                    f1 = ET.Element("bpmn:sequenceFlow", {"id": f1_id, "sourceRef": prev_id, "targetRef": bridge_id})
                    process_node.append(f1)
                    e1 = ET.Element("bpmndi:BPMNEdge", {"id": f"{f1_id}_di", "bpmnElement": f1_id})
                    w1_1 = ET.Element("omgdi:waypoint", {"x": str(prev_x), "y": str(prev_y)})
                    w1_2 = ET.Element("omgdi:waypoint", {"x": str(mid_x), "y": str(mid_y + 40)})
                    e1.append(w1_1)
                    e1.append(w1_2)
                    plane_node.append(e1)

                    # 连线 2
                    f2_id = f"Flow_B2C_{i}"
                    f2 = ET.Element("bpmn:sequenceFlow", {"id": f2_id, "sourceRef": bridge_id, "targetRef": first_node})
                    process_node.append(f2)
                    e2 = ET.Element("bpmndi:BPMNEdge", {"id": f"{f2_id}_di", "bpmnElement": f2_id})
                    w2_1 = ET.Element("omgdi:waypoint", {"x": str(mid_x + 100), "y": str(mid_y + 40)})
                    w2_2 = ET.Element("omgdi:waypoint", {"x": str(curr_left_x), "y": str(curr_center_y)})
                    e2.append(w2_1)
                    e2.append(w2_2)
                    plane_node.append(e2)

                    print(f"    -> 连接: {prev_id} -> [{bridge_text}] -> {first_node}")

            # 更新状态
            if last_node:
                geo = self._get_node_center(atom_root, last_node)
                if geo:
                    x, y, w, h = geo
                    prev_last_node_info = (last_node, x + w, y + h / 2)

            width = max_x - min_x
            global_cursor_x += (width + 250.0)

        if not os.path.exists(OUTPUT_XML_DIR): os.makedirs(OUTPUT_XML_DIR)
        output_path = os.path.join(OUTPUT_XML_DIR, f"{complex_data.get('id')}.bpmn")
        tree = ET.ElementTree(merged_root)
        tree.write(output_path, encoding='utf-8', xml_declaration=True)
        print(f"✅ 生成完毕: {output_path}\n")


if __name__ == "__main__":
    merger = BpmnMerger()
    if os.path.exists(COMPLEX_DATA_DIR):
        for f in os.listdir(COMPLEX_DATA_DIR):
            if f.startswith("complex_") and f.endswith(".json"):
                merger.merge(os.path.join(COMPLEX_DATA_DIR, f))
import asyncio
import importlib.util
import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COMFY = ROOT.parents[1]
IDS = {"CurveAttnSparse": "CurveAttn Sparse", "CurveAttnCache": "CurveAttn Residual Cache"}


class RegistrationTests(unittest.TestCase):
    def load(self):
        name = "curvattn_real_registration_test"
        spec = importlib.util.spec_from_file_location(
            name, ROOT / "__init__.py", submodule_search_locations=[str(ROOT)])
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        self.addCleanup(lambda: [sys.modules.pop(k) for k in tuple(sys.modules)
                                 if k == name or k.startswith(name + ".")])
        spec.loader.exec_module(module)
        return module

    def test_v3_only(self):
        module = self.load()
        self.assertFalse(hasattr(module, "NODE_CLASS_MAPPINGS"))
        self.assertFalse(hasattr(module, "NODE_DISPLAY_NAME_MAPPINGS"))
        self.assertEqual(module.__version__, "3.0.0")

        async def listed():
            return await (await module.comfy_entrypoint()).get_node_list()
        nodes = asyncio.run(listed())
        self.assertEqual(nodes, [module.CurveAttnSparse, module.CurveAttnCache])
        for node in nodes:
            self.assertNotIn("INPUT_TYPES", node.__dict__)
            schema = node.define_schema()
            schema.validate()
            self.assertEqual(schema.display_name, IDS[schema.node_id])
            info = node.GET_NODE_INFO_V1()
            self.assertEqual(info["name"], schema.node_id)
            self.assertIn("MODEL", info["output"])

    def test_no_legacy_names_or_mappings_in_sources(self):
        for path in ROOT.rglob("*.py"):
            if "tests" in path.parts:
                continue
            text = path.read_text(encoding="utf-8")
            self.assertNotIn("NODE_CLASS_MAPPINGS", text, path)
            self.assertNotRegex(text, r"node_id=\"CurveAttention\"", path)
            self.assertNotIn('getLogger("CurveAttention")', text, path)

    def test_node_ids_unique_in_core_and_custom_nodes(self):
        pattern = re.compile(r"(CurveAttnSparse|CurveAttnCache)\b")
        roots = [COMFY / "comfy_extras", COMFY / "comfy_api_nodes"]
        files = [COMFY / "nodes.py"] + [p for r in roots if r.is_dir() for p in r.rglob("*.py")]
        for pack in (COMFY / "custom_nodes").iterdir():
            if pack.is_dir() and pack.resolve() != ROOT.resolve():
                files += list(pack.glob("*.py")) + list(pack.glob("*/*.py"))
        clashes = [str(p) for p in files if p.is_file()
                   and pattern.search(p.read_text(encoding="utf-8", errors="ignore"))]
        self.assertEqual(clashes, [])

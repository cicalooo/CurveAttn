import asyncio
import importlib.util
import sys
import unittest
from pathlib import Path


class RegistrationTests(unittest.TestCase):
    def test_v3_only(self):
        root = Path(__file__).resolve().parents[1]
        name = "curve_real_registration_test"
        spec = importlib.util.spec_from_file_location(
            name, root / "__init__.py", submodule_search_locations=[str(root)])
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
            self.assertFalse(hasattr(module, "NODE_CLASS_MAPPINGS"))
            self.assertNotIn("INPUT_TYPES", module.CurveAttention.__dict__)
            async def listed():
                extension = await module.comfy_entrypoint()
                return await extension.get_node_list()
            self.assertEqual(asyncio.run(listed()), [module.CurveAttention])
            schema = module.CurveAttention.define_schema()
            self.assertEqual(schema.node_id, "CurveAttention")
        finally:
            for key in tuple(sys.modules):
                if key == name or key.startswith(name + "."):
                    del sys.modules[key]

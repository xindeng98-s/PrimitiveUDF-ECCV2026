import unittest
from unittest.mock import patch
import torch
from heads import primitive_geometry as geometry
from heads.primitive_bank_v3 import PrimitiveBankV3Head

def dense_knn(*, k, sup_xyz, sup_offset, qry_xyz, qry_offset):
    out = torch.full((len(qry_xyz), k), -1, dtype=torch.long)
    s = q = 0
    for se, qe in zip(sup_offset.tolist(), qry_offset.tolist()):
        n = min(k, se-s)
        if qe > q and n:
            out[q:qe, :n] = torch.cdist(qry_xyz[q:qe], sup_xyz[s:se]).topk(n, largest=False).indices+s
        s, q = se, qe
    return out

class V3Tests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(41)
        torch.set_num_threads(2)
        self.head = PrimitiveBankV3Head(primitive_geometry_backend='torch')
        self.x = .1 * torch.randn(40,3)
        self.features = torch.randn(40,48,requires_grad=True)
        self.q = (.05*torch.randn(8,3)).requires_grad_()
        self.so, self.qo = torch.tensor([20,40]), torch.tensor([4,8])

    @patch.object(geometry, 'knn_query_dense', side_effect=dense_knn)
    def test_normal_loss_second_backward(self, _):
        y,_ = self.head((self.x,self.features,self.so),self.q,self.qo)
        g = torch.autograd.grad(y.sum(),self.q,create_graph=True)[0]
        (y.square().mean()+g.square().mean()).backward()
        self.assertTrue(torch.isfinite(g).all())
        self.assertTrue(torch.isfinite(self.features.grad).all())
        self.assertTrue(torch.isfinite(self.head.outs.weight.grad).all())

    @patch.object(geometry, 'knn_query_dense', side_effect=dense_knn)
    def test_cache_preserves_output_and_batch_isolation(self, _):
        h = self.head.eval()
        with torch.no_grad():
            bank = h.build_bank((self.x,self.features,self.so))
            y,_ = h.predict(bank,self.q,self.qo)
            self.assertTrue(torch.isfinite(y).all())
            h.interpolators.cache_point_features = False
            direct,_ = h((self.x,self.features,self.so),self.q,self.qo)
            torch.testing.assert_close(y,direct,rtol=1e-5,atol=1e-6)
            neighbors,_,_,_ = h.interpolators._acquire(bank[1],self.q,self.qo)
            self.assertTrue((neighbors[:4] < 20).all())
            self.assertTrue((neighbors[4:] >= 20).all())

    def test_reject_approximation(self):
        with self.assertRaises(ValueError):
            PrimitiveBankV3Head(primitive_topology_voxel_size=.01)

class MeshExtractionTests(unittest.TestCase):
    def test_unsigned_sphere_extracts_faces(self):
        import numpy as np
        from networks.custom_mc import udf_mc_lewiner
        a = np.linspace(-.6, .6, 32, dtype='float32')
        xyz = np.stack(np.meshgrid(a, a, a, indexing='ij'), axis=-1)
        radius = np.linalg.norm(xyz, axis=-1)
        signed = radius - .3
        direction = -xyz / radius[..., None] * np.sign(signed)[..., None]
        vertices, faces, _, _ = udf_mc_lewiner(abs(signed), direction.astype('float32'))
        self.assertGreater(len(faces), 0)
        self.assertTrue(np.isfinite(vertices).all())

if __name__ == '__main__':
    unittest.main()

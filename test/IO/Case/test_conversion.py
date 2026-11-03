"""转换器的语义回归与可选CUDA端到端检查，不依赖外部Python包。"""
import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace

PROJECT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT / "scripts"))
from caseconvert import elmer, inp
from caseconvert.common import ConversionError, constant, stepLimit, writeBundle


TET = """*NODE,NSET=ALL
1,0,0,0
2,1,0,0
3,0,1,0
4,0,0,1
*ELEMENT,TYPE=DC3D4,ELSET=BODY
10,1,2,3,4
*NSET,NSET=COLD
1,2,3
*MATERIAL,NAME=SOLID
*CONDUCTIVITY
2
*DENSITY
3
*SPECIFIC HEAT
4
*SOLID SECTION,ELSET=BODY,MATERIAL=SOLID
*STEP
*HEAT TRANSFER,STEADY STATE
*BOUNDARY
COLD,11,11,300
*DFLUX
BODY,BF,12
*END STEP
"""

SIF = """Header
Mesh DB "." "mesh"
End
Simulation
Coordinate System = "Cartesian 2D"
Simulation Type = Steady State
End
Body 1
Equation = 1
Material = 1
Body Force = 1
End
Equation 1
Active Solvers(1) = 1
End
Solver 1
Procedure = "HeatSolve" "HeatSolver"
Variable = Temperature
End
Material 1
Heat Conductivity = 2
Density = 3
Heat Capacity = 4
End
Body Force 1
Heat Source = 4
End
Boundary Condition 1
Target Boundaries = 1
Temperature = Real 300
End
Boundary Condition 2
Target Boundaries = 2
Heat Flux BC = True
Heat Flux = 5
End
"""


def options(**kwargs):
    result = dict(temperatureUnit="K", thickness=1., meshDir=None,
                  initialTemperature=None, backwardEulerStep=None)
    result.update(kwargs)
    return SimpleNamespace(**result)


class ConversionTests(unittest.TestCase):
    def test_time_budget_rejects_overflow(self):
        self.assertEqual(stepLimit(.2, .1), 3)
        for duration, step in ((1., 0.), (1., 1e-320), (1e300, 1e-300)):
            with self.assertRaises(ConversionError):
                stepLimit(duration, step)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def deck(self, text=TET):
        path = self.root / "source.inp"
        path.write_text(text, encoding="utf-8")
        return path

    def sif(self, text=SIF):
        path = self.root / "source.sif"
        path.write_text(text, encoding="utf-8")
        mesh = self.root / "mesh"
        mesh.mkdir(exist_ok=True)
        for name, value in {
            "header": "3 1 3\n2\n303 1\n202 3\n",
            "nodes": "1 -1 0 0 0\n2 -1 1 0 0\n3 -1 0 1 0\n",
            "elements": "1 1 303 1 2 3\n",
            "boundary": "1 1 1 0 202 1 2\n2 2 1 0 202 2 3\n3 3 1 0 202 3 1\n",
        }.items():
            (mesh / f"mesh.{name}").write_text(value)
        return path

    def test_inp_volume_source_and_exact_constraint_faces(self):
        model = inp.convert(self.deck(), options(thickness=None))
        model.validate()
        self.assertEqual(model.sources, [({10}, 12.)])
        self.assertEqual(model.boundaries[0][0], {(1, 2, 3)})
        self.assertEqual(model.materials['SOLID']['conductivity'], 2)

    def test_inp_surface_flux_sign_and_tetra_face_number(self):
        text = TET.replace('BODY,BF,12', 'BODY,S3,5\n').replace('*END STEP', '*FILM\nBODY,F4,290,7\n*END STEP')
        model = inp.convert(self.deck(text), options(thickness=None))
        model.validate()
        self.assertEqual(model.boundaries[1][0], {(2, 3, 4)})
        self.assertEqual(model.boundaries[1][1]['flux'], constant(-5))
        self.assertEqual(model.boundaries[2][0], {(1, 3, 4)})
        self.assertEqual(model.boundaries[2][1]['ambientTemperature'], constant(290))

    def test_inp_include_relative_nested_sets_and_generate(self):
        self.deck().rename(self.root / 'physics.inp')
        path = self.deck('*INCLUDE,INPUT=physics.inp\n')
        model = inp.convert(path, options(thickness=None))
        self.assertEqual(len(model.inputs), 2)
        text = TET.replace('*NSET,NSET=COLD\n1,2,3', '*NSET,NSET=A,GENERATE\n1,3,1\n*NSET,NSET=COLD\nA')
        model = inp.convert(self.deck(text), options(thickness=None))
        self.assertEqual(model.boundaries[0][0], {(1, 2, 3)})

    def test_inp_celsius_conversion(self):
        model = inp.convert(self.deck(TET.replace('11,11,300', '11,11,20')), options(thickness=None, temperatureUnit='C'))
        self.assertAlmostEqual(model.boundaries[0][1]['temperature']['value'], 293.15)

    def test_inp_transient_requires_explicit_discretization_and_initial_state(self):
        text = TET.replace('*STEP', '*INITIAL CONDITIONS,TYPE=TEMPERATURE\nALL,300\n*STEP')
        text = text.replace('*HEAT TRANSFER,STEADY STATE', '*HEAT TRANSFER,DIRECT\n0.1,0.2')
        with self.assertRaises(ConversionError):
            inp.convert(self.deck(text), options(thickness=None))
        model = inp.convert(self.deck(text), options(thickness=None, backwardEulerStep=.1))
        model.validate()
        self.assertEqual(model.time['end'], .2)
        self.assertEqual(model.initial, {10: 300})

    def test_inp_refuses_physics_that_cannot_be_preserved(self):
        cases = [TET.replace('DC3D4', 'C3D10'), TET.replace('*CONDUCTIVITY\n2', '*CONDUCTIVITY,TYPE=ORTHO\n2'),
                 TET.replace('*CONDUCTIVITY\n2', '*CONDUCTIVITY\n2,300\n3,400'),
                 TET.replace('*DFLUX', '*DFLUX,AMPLITUDE=TIME'), TET.replace('*DFLUX', '*RADIATE'),
                 TET.replace('1,2,3\n*MATERIAL', '1\n*MATERIAL'), TET.replace('11,11,300', '1,1,300'),
                 TET.replace('*END STEP', '*END STEP\n*STEP\n*HEAT TRANSFER\n*END STEP'),
                 TET.replace('*NSET,NSET=COLD\n1,2,3', '*NSET,NSET=COLD\nOTHER\n*NSET,NSET=OTHER\nCOLD')]
        for text in cases:
            with self.subTest(text=text[-100:]), self.assertRaises(ConversionError):
                model = inp.convert(self.deck(text), options(thickness=None))
                model.validate()

    def test_inp_include_cycle(self):
        with self.assertRaises(ConversionError):
            inp.convert(self.deck('*INCLUDE,INPUT=source.inp'), options(thickness=None))

    def test_inp_two_dimensional_thickness_and_edge_number(self):
        text = TET.replace('4,0,0,1\n', '').replace('DC3D4', 'DC2D3').replace('10,1,2,3,4', '10,1,2,3')
        text = text.replace('1,2,3\n*MATERIAL', '1,2\n*MATERIAL').replace('*STEP', '0.25\n*STEP')
        text = text.replace('BODY,BF,12', 'BODY,S2,5')
        model = inp.convert(self.deck(text), options(thickness=None))
        model.validate()
        self.assertEqual(model.thickness, .25)
        self.assertEqual(model.boundaries[1][0], {(2, 3)})

    def test_reversed_orientation_preserves_source_face_identity(self):
        text = TET.replace('10,1,2,3,4', '10,2,1,3,4').replace('BODY,BF,12', 'BODY,S2,5')
        model = inp.convert(self.deck(text), options(thickness=None))
        model.validate()
        self.assertEqual(model.cells[10], (1, 2, 3, 4))
        self.assertEqual(model.boundaries[1][0], {(1, 2, 4)})

    def test_elmer_mass_source_flux_and_body_mapping(self):
        model = elmer.convert(self.sif(), options())
        model.validate()
        self.assertEqual(model.sources, [({1}, 12.)])
        self.assertEqual(model.boundaries[1][1]['flux'], constant(-5))
        self.assertEqual(model.assignments, {'material-1': {1}})

    def test_elmer_convection_and_volumetric_source(self):
        text = SIF.replace('Heat Source = 4', 'Volumetric Heat Source = 4')
        text = text.replace('Heat Flux = 5', 'Heat Transfer Coefficient = 7\nExternal Temperature = 290')
        model = elmer.convert(self.sif(text), options())
        model.validate()
        self.assertEqual(model.sources[0][1], 4)
        self.assertEqual(model.boundaries[1][1]['coefficient'], constant(7))

    def test_elmer_time_integrator_is_not_silently_changed(self):
        text = SIF.replace('Steady State', 'Transient\nTimestepping Method = BDF\nBDF Order = 2\nTimestep Sizes = 0.1\nTimestep Intervals = 2')
        with self.assertRaises(ConversionError):
            elmer.convert(self.sif(text), options(initialTemperature=300))
        model = elmer.convert(self.sif(text), options(initialTemperature=300, backwardEulerStep=.05))
        model.validate()
        self.assertEqual(model.time['end'], .2)
        self.assertEqual(model.time['step'], .05)

    def test_elmer_rejects_unknown_physics_expressions_and_noncartesian(self):
        for text in [SIF.replace('Heat Conductivity = 2', 'Heat Conductivity = Variable Temperature'),
                     SIF.replace('Heat Source = 4', 'Heat Source = 4\nConvection Velocity 1 = 1'),
                     SIF.replace('Cartesian 2D', 'Axi Symmetric'), SIF.replace('Heat Flux = 5', 'Radiation = True'),
                     SIF.replace('Heat Flux BC = True\n', ''),
                     SIF.replace('Heat Conductivity = 2', 'Heat Conductivity = 2\nHeat Conductivity = 3')]:
            with self.subTest(text=text[-80:]), self.assertRaises(ConversionError):
                elmer.convert(self.sif(text), options())

    def test_elmer_header_and_connectivity_validation(self):
        path = self.sif()
        (self.root / 'mesh/mesh.header').write_text('4 1 3\n2\n303 1\n202 3\n')
        with self.assertRaises(ConversionError):
            elmer.convert(path, options())

    def test_elmer_three_dimensional_mesh_and_zero_parent_metadata(self):
        path = self.sif(SIF.replace('Cartesian 2D', 'Cartesian 3D'))
        for name, value in {
            'header': '4 1 4\n2\n504 1\n303 4\n',
            'nodes': '1 -1 0 0 0\n2 -1 1 0 0\n3 -1 0 1 0\n4 -1 0 0 1\n',
            'elements': '1 1 504 1 2 3 4\n',
            'boundary': '1 1 0 0 303 1 2 3\n2 2 1 0 303 2 3 4\n3 3 1 0 303 1 3 4\n4 3 1 0 303 1 2 4\n',
        }.items():
            (self.root/'mesh'/f'mesh.{name}').write_text(value)
        model = elmer.convert(path, options(thickness=None))
        model.validate()
        self.assertEqual(model.dimension, 3)
        self.assertEqual(model.boundaries[0][0], {(1, 2, 3)})

    def test_bundle_publish_hashes_overlap_and_no_overwrite(self):
        model = inp.convert(self.deck(), options(thickness=None))
        model.sources.append(({10}, 1.))
        output = writeBundle(model, self.root / 'converted', 'inp', 'test')
        case = json.loads((output / 'case.json').read_text())
        report = json.loads((output / 'conversion-report.json').read_text(encoding='utf-8'))
        self.assertEqual(len(case['sources']), 2)
        self.assertEqual(len(report['inputs'][0]['sha256']), 64)
        self.assertIn('$MeshFormat\n4.1', (output / 'mesh.msh').read_text())
        with self.assertRaises(ConversionError):
            writeBundle(model, output, 'inp', 'test')
        model.materials['SOLID']['conductivity'] = -1
        with self.assertRaises(ConversionError):
            writeBundle(model, self.root / 'invalid', 'inp', 'test')
        self.assertFalse((self.root / 'invalid').exists())

    def test_bundle_rejects_degenerate_duplicate_and_conflicting_conditions(self):
        model = inp.convert(self.deck(), options(thickness=None))
        for mutate in [lambda m: m.nodes.update({4: m.nodes[1]}),
                       lambda m: m.cells.update({11: m.cells[10]}),
                       lambda m: m.boundaries.append(({(1, 2, 3)}, {'kind': 'outwardFlux', 'flux': constant(1)}))]:
            candidate = copy.deepcopy(model)
            mutate(candidate)
            with self.assertRaises(ConversionError):
                candidate.validate()


def integration(executable):
    """通过真实CLI加载转换产物，检查体源/输入热流/对流的解析答案及瞬态温升。"""
    with tempfile.TemporaryDirectory(prefix='emberfield-conversion-') as directory:
        root = Path(directory)
        cases = [
            ('tet-source', TET, options(thickness=None), 301.5),
            ('tet-flux', TET.replace('BODY,BF,12', 'BODY,S3,2'), options(thickness=None), 300+3**.5),
            ('tet-film', TET.replace('BODY,BF,12', '').replace('*END STEP', '*FILM\nBODY,F3,310,2\n*END STEP'),
             options(thickness=None), 300 + (10/(3**.5)) / (1/3 + (3**.5)/6)),
        ]
        for name, text, args, expected in cases:
            path = root / f'{name}.inp'
            path.write_text(text)
            model = inp.convert(path, args)
            bundle = writeBundle(model, root / name, 'inp', name)
            result = subprocess.run([str(executable), str(bundle / 'case.json'), '--output-root', str(root),
                                     '--output', name+'-result'], capture_output=True, text=True)
            require_message = (result.returncode, result.stdout, result.stderr)
            assert result.returncode == 0, require_message
            summary = json.loads((root / (name+'-result') / 'run-summary.json').read_text())
            assert summary['status'] == 'completed', summary
            import csv
            with (root / (name+'-result') / 'steps.csv').open() as stream:
                row = list(csv.DictReader(stream))[-1]
            assert abs(float(row['temperature_max_k']) - expected) < 1e-7, (name, row, expected)
        # 空间常量温升：无外向通量，Q/(rho*cp)=1 K/s。
        transient = TET.replace('*BOUNDARY\nCOLD,11,11,300\n', '')
        transient = transient.replace('*STEP', '*INITIAL CONDITIONS,TYPE=TEMPERATURE\nALL,300\n*STEP')
        transient = transient.replace('*HEAT TRANSFER,STEADY STATE', '*HEAT TRANSFER,DIRECT\n0.1,0.2')
        path = root / 'transient.inp'
        path.write_text(transient)
        model = inp.convert(path, options(thickness=None, backwardEulerStep=.1))
        bundle = writeBundle(model, root / 'transient', 'inp', 'transient')
        result = subprocess.run([str(executable), str(bundle/'case.json'), '--output-root', str(root), '--output', 'transient-result'],
                                capture_output=True, text=True)
        assert result.returncode == 0, (result.stdout, result.stderr)
        with (root/'transient-result/steps.csv').open() as stream:
            row = list(csv.DictReader(stream))[-1]
        assert abs(float(row['temperature_max_k']) - 300.2) < 1e-7, row
        # Elmer路径的质量热源与斜边输入热流合成解析答案。
        helper = ConversionTests()
        helper.root = root
        model = elmer.convert(helper.sif(), options())
        bundle = writeBundle(model, root/'elmer-triangle', 'elmer', 'triangle')
        result = subprocess.run([str(executable), str(bundle/'case.json'), '--output-root', str(root), '--output', 'triangle-result'],
                                capture_output=True, text=True)
        assert result.returncode == 0, (result.stdout, result.stderr)
        with (root/'triangle-result/steps.csv').open() as stream:
            row = list(csv.DictReader(stream))[-1]
        assert abs(float(row['temperature_max_k']) - (302+5*2**.5/2)) < 1e-7, row
        print('Converted source, flux, convection and transient analytic solutions: passed')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--executable', type=Path)
    args, remaining = parser.parse_known_args()
    result = unittest.main(argv=[sys.argv[0], *remaining], exit=False)
    if not result.result.wasSuccessful():
        sys.exit(1)
    if args.executable:
        integration(args.executable.resolve())

"""使用独立Gmsh 4.14.1交叉验证网格读写；依赖仅用于可选验证，不属于运行库。"""
import argparse
import math
import pathlib
import subprocess
import shutil
import tempfile

import gmsh


def snapshot():
    """按持久标签取得节点、单元、物理组与分类，不依赖数组顺序。"""
    tags, xyz, _ = gmsh.model.mesh.getNodes()
    nodes = {int(tag): tuple(float(x) for x in xyz[3*i:3*i+3]) for i, tag in enumerate(tags)}
    cells = {}
    for kind, tags, connectivity in zip(*gmsh.model.mesh.getElements()):
        arity = gmsh.model.mesh.getElementProperties(int(kind))[3]
        for i, tag in enumerate(tags):
            cells[int(tag)] = (int(kind), tuple(int(x) for x in connectivity[i*arity:(i+1)*arity]))
    entities = {}
    for dim, tag in gmsh.model.getEntities():
        _, etags, _ = gmsh.model.mesh.getElements(dim, tag)
        cells_here = tuple(sorted(int(x) for group in etags for x in group))
        groups = tuple(sorted(int(x) for x in gmsh.model.getPhysicalGroupsForEntity(dim, tag)))
        parts = tuple(int(x) for x in gmsh.model.getPartitions(dim, tag))
        parent = gmsh.model.getParent(dim, tag)
        entities[(dim, tag)] = (cells_here, groups, parts, parent)
    physical = {(dim, tag): gmsh.model.getPhysicalName(dim, tag) for dim, tag in gmsh.model.getPhysicalGroups()}
    return nodes, cells, entities, physical


def compare(expected, actual):
    """比较完整连接次序及可访问元数据，坐标使用浮点容差。"""
    assert expected[0].keys() == actual[0].keys(), "node identities changed"
    for tag, coordinates in expected[0].items():
        assert all(math.isclose(a, b, rel_tol=1e-13, abs_tol=1e-14) for a, b in zip(coordinates, actual[0][tag])), tag
    assert expected[1:] == actual[1:], "connectivity/classification/groups/partitions changed"


def generate(dimension, partitioned, periodic):
    """生成高阶二维或三维网格，包含边界实体与物理组。"""
    gmsh.clear()
    gmsh.model.add("interop")
    if dimension == 3:
        volume = gmsh.model.occ.addBox(0, 0, 0, 1, 1, 1)
        gmsh.model.occ.synchronize()
        gmsh.model.addPhysicalGroup(3, [volume], 9, "volume")
    else:
        points = [gmsh.model.geo.addPoint(x, y, 0, 0.4) for x, y in [(0, 0), (1, 0), (1, 1), (0, 1)]]
        edges = [gmsh.model.geo.addLine(points[i], points[(i+1) % 4]) for i in range(4)]
        loop = gmsh.model.geo.addCurveLoop(edges)
        surface = gmsh.model.geo.addPlaneSurface([loop])
        gmsh.model.geo.synchronize()
        gmsh.model.addPhysicalGroup(2, [surface], 8, "domain")
        gmsh.model.addPhysicalGroup(1, [edges[0], edges[2]], 5, "boundary")
        if periodic:
            gmsh.model.mesh.setPeriodic(1, [edges[1]], [edges[3]], [1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1])
    gmsh.option.setNumber("Mesh.MeshSizeMax", 0.45)
    gmsh.model.mesh.generate(dimension)
    gmsh.model.mesh.setOrder(2)
    if partitioned:
        gmsh.option.setNumber("Mesh.PartitionCreateGhostCells", 1)
        gmsh.model.mesh.partition(2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("converter", type=pathlib.Path)
    args = parser.parse_args()
    converter = args.converter.resolve()
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    try:
        with tempfile.TemporaryDirectory(prefix="thermofem-gmsh-") as directory:
            directory = pathlib.Path(directory)
            cases = [(2, False, False), (2, False, True), (2, True, False), (3, False, False)]
            count = 0
            for dimension, partitioned, periodic in cases:
                for version in [4.1, 2.2] if not partitioned else [4.1]:
                    for binary in [0, 1]:
                        generate(dimension, partitioned, periodic)
                        gmsh.option.setNumber("Mesh.MshFileVersion", version)
                        gmsh.option.setNumber("Mesh.SaveAll", 1)
                        gmsh.option.setNumber("Mesh.SaveParametric", int(version == 4.1))
                        gmsh.option.setNumber("Mesh.Binary", binary)
                        source = directory / "source.msh"
                        gmsh.write(str(source))
                        gmsh.clear()
                        gmsh.open(str(source))
                        expected = snapshot()
                        for encoding in ["ascii", "binary"]:
                            target = directory / (encoding + ".msh")
                            subprocess.run([str(converter), str(source), str(target), encoding, str(version)], check=True, capture_output=True, text=True)
                            gmsh.clear()
                            try:
                                gmsh.open(str(target))
                            except Exception:
                                shutil.copy(source, "build/gmsh-source-failed.msh")
                                shutil.copy(target, "build/gmsh-target-failed.msh")
                                print("Failed case:", dimension, partitioned, periodic, version, binary, encoding)
                                raise
                            compare(expected, snapshot())
                            count += 1
            print(f"Gmsh {gmsh.__version__}: {count} independent cross-read/write checks passed")
    finally:
        gmsh.finalize()


if __name__ == "__main__":
    main()

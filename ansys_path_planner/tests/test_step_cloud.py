"""STEP 转点云测试。"""

import numpy as np

from path_planner.parsers.step_cloud import parse_step_file, parse_step_points


_BOX_STEP = """ISO-10303-21;
HEADER;
FILE_DESCRIPTION((''),'2;1');
FILE_NAME('box','2024-01-01T00:00:00',(''),(''),'','','','');
FILE_SCHEMA(('AUTOMOTIVE_DESIGN { 1 0 10303 214 1 1 1 1 }'));
ENDSEC;
DATA;
#1=CARTESIAN_POINT('p0',(0.,0.,0.));
#2=CARTESIAN_POINT('p1',(10.,0.,0.));
#3=CARTESIAN_POINT('p2',(10.,10.,0.));
#4=CARTESIAN_POINT('p3',(0.,10.,0.));
#5=DIRECTION('',(0.,0.,1.));
#6=DIRECTION('',(1.,0.,0.));
#7=AXIS2_PLACEMENT_3D('',#1,#5,#6);
#8=PLANE('',#7);
#9=VERTEX_POINT('',#1);
#10=VERTEX_POINT('',#2);
#11=VERTEX_POINT('',#3);
#12=VERTEX_POINT('',#4);
#13=LINE('',#1,#6);
#14=LINE('',#2,(0.,1.,0.));
#15=LINE('',#3,(-1.,0.,0.));
#16=LINE('',#4,(0.,-1.,0.));
#17=EDGE_CURVE('',#9,#10,#13,.T.);
#18=EDGE_CURVE('',#10,#11,#14,.T.);
#19=EDGE_CURVE('',#11,#12,#15,.T.);
#20=EDGE_CURVE('',#12,#9,#16,.T.);
#21=ORIENTED_EDGE('',*,*,#17,.T.);
#22=ORIENTED_EDGE('',*,*,#18,.T.);
#23=ORIENTED_EDGE('',*,*,#19,.T.);
#24=ORIENTED_EDGE('',*,*,#20,.T.);
#25=EDGE_LOOP('',(#21,#22,#23,#24));
#26=FACE_OUTER_BOUND('',#25,.T.);
#27=ADVANCED_FACE('',(#26),#8,.T.);
ENDSEC;
END-ISO-10303-21;
"""


def test_parse_step_points(tmp_path):
    path = tmp_path / "box.step"
    path.write_text(_BOX_STEP, encoding="utf-8")
    pts = parse_step_points(str(path))
    # 顶点 + 平面面采样应生成大量点
    assert len(pts) > 50
    lo = pts.min(axis=0)
    hi = pts.max(axis=0)
    assert hi[0] <= 10.001
    assert hi[1] <= 10.001
    assert abs(hi[2]) < 1e-6


def test_parse_step_file_dataframe(tmp_path):
    path = tmp_path / "box.step"
    path.write_text(_BOX_STEP, encoding="utf-8")
    df = parse_step_file(str(path))
    assert list(df.columns) == ["Node", "X", "Y", "Z"]
    assert len(df) > 50
    assert df["Node"].is_unique

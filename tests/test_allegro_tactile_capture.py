import ast
from pathlib import Path
from threading import Lock, Event
from types import SimpleNamespace
import time, os, json
import numpy as np

SOURCE = Path(__file__).resolve().parents[1] / 'paradex/io/robot_controller/allegro_v5_controller_ros2.py'
if not SOURCE.exists(): SOURCE = Path(__file__).with_name('controller.py')

def load():
    tree=ast.parse(SOURCE.read_text())
    nodes=[n for n in tree.body if isinstance(n,ast.ClassDef)]
    ns=dict(np=np,time=time,os=os,json=json,Lock=Lock,Event=Event,Node=object,Int32MultiArray=object,JointState=object,
            action_dof=16,RECV_MIN_INTERVAL=.01,TACTILE_SUFFIX='/allegroHand_0/tactile_sensors')
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(SOURCE),'exec'),ns)
    return ns

def fixture():
    ns=load();shared=ns['_Shared']();cls=ns['AllegroController']
    controller=object.__new__(cls);controller._shared=shared;controller.tactile_enabled=True;controller.namespace='right'
    node=object.__new__(ns['_StateReceiverNode']);node._shared=shared;node._last_tactile_t=0.
    return controller,node

def test_packets_record_even_when_command_publishing_is_idle_and_fast(tmp_path):
    c,n=fixture();c.start(str(tmp_path))
    for i in range(10): n._tactile_cb(SimpleNamespace(data=[i]*16))
    c.stop()
    raw=np.load(tmp_path/'tactile_raw.npy',allow_pickle=False)
    ts=np.load(tmp_path/'tactile_time.npy',allow_pickle=False)
    assert raw.shape==(10,16) and raw.dtype==np.int32
    assert ts.shape==(10,) and np.all(np.diff(ts)>=0) and abs(ts[-1]-time.time())<5
    assert len(np.load(tmp_path/'time.npy'))==0
    assert json.loads((tmp_path/'tactile_info.json').read_text())['samples']==10

def test_episode_boundaries_and_missing_packets(tmp_path,capsys):
    c,n=fixture();n._tactile_cb(SimpleNamespace(data=[3]*16))
    c.start(str(tmp_path/'first'));n._tactile_cb(SimpleNamespace(data=[4]*16));c.stop()
    n._tactile_cb(SimpleNamespace(data=[5]*16))
    c.start(str(tmp_path/'second'));c.stop()
    assert np.load(tmp_path/'first/tactile_raw.npy').shape==(1,16)
    assert np.load(tmp_path/'second/tactile_time.npy').size==0
    assert 'no Allegro tactile' in capsys.readouterr().out

def test_malformed_packets_do_not_break_saving(tmp_path):
    c,n=fixture();c.start(str(tmp_path));n._tactile_cb(SimpleNamespace(data=[1]*16));n._tactile_cb(SimpleNamespace(data=[]));n._tactile_cb(SimpleNamespace(data=[1]*8));c.stop()
    assert np.load(tmp_path/'tactile_raw.npy').shape==(1,16)
    assert json.loads((tmp_path/'tactile_info.json').read_text())['malformed_packets']==2

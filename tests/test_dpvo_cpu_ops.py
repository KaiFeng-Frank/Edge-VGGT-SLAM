"""Tiny synthetic CPU-only DPVO operation oracles; no images or weights."""
import importlib.util
import itertools
import math
import struct
from pathlib import Path

import pytest
import torch

HERE = Path(__file__).resolve().parent
IMPL = HERE if (HERE / 'dpvo_cpu_ops.py').is_file() else HERE.parent / 'deploy/rk3588/validation'
spec = importlib.util.spec_from_file_location("dpvo_cpu_ops", IMPL / 'dpvo_cpu_ops.py')
ops = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ops)


def neighbor_oracle(kk, jj):
    prev, nxt = [-1] * len(kk), [-1] * len(kk)
    for group in sorted(set(kk)):
        rows = [i for i, key in enumerate(kk) if key == group]
        rows.sort(key=lambda i: jj[i])  # Python stable sort is independent of Torch sort.
        for p, row in enumerate(rows):
            prev[row] = rows[p - 1] if p else -1
            nxt[row] = rows[p + 1] if p + 1 < len(rows) else -1
    return torch.tensor(prev), torch.tensor(nxt)


@pytest.mark.parametrize("kk,jj", [([], []), ([5], [2]), ([4,4,4,7,7,4], [3,1,1,8,7,2]), ([-1,0,-1,0], [4,4,2,4]), ([2**54,2**54+1,2**54], [3,0,1])])
def test_neighbors_stable_groups_and_original_order(kk, jj):
    args = [torch.tensor(v, dtype=torch.int64) for v in (kk,jj)]
    actual = ops.neighbors(*args)
    expected = neighbor_oracle(kk,jj)
    assert all(x.dtype == torch.int64 and x.device.type == "cpu" for x in actual)
    assert all(torch.equal(x, y.to(torch.int64)) for x,y in zip(actual,expected))


def scatter_oracle(src, index, dim, softmax):
    dim %= src.ndim
    shape = list(src.shape)
    shape[dim] = src.shape[dim] if softmax else int(index.max()) + 1 if index.numel() else 0
    result = src.new_zeros(shape)
    outer_shape = list(src.shape)
    outer_shape.pop(dim)
    for outer in itertools.product(*(range(d) for d in outer_shape)):
        def location(i):
            return outer[:dim] + (i,) + outer[dim:]
        for key in set(index.tolist()):
            members = [i for i, group in enumerate(index.tolist()) if group == key]
            values = [float(src[location(i)]) for i in members]
            if softmax:
                exps = [math.exp(value-max(values)) for value in values]
                for i, value in zip(members, exps):
                    result[location(i)] = value/sum(exps)
            else:
                result[location(key)] = sum(values)
    return result


@pytest.mark.parametrize("dim,shape", [(0,(5,2,3)), (1,(2,5,3)), (-1,(2,3,5))])
@pytest.mark.parametrize("softmax", [False,True])
def test_scatter_unsorted_gaps_per_channel_and_large_scores(dim,shape,softmax):
    src = torch.arange(math.prod(shape), dtype=torch.float64).reshape(shape)*.2+10000
    index = torch.tensor([3,0,3,1,0])
    function = ops.scatter_softmax if softmax else ops.scatter_sum
    actual = function(src,index,dim)
    expected = scatter_oracle(src,index,dim,softmax)
    torch.testing.assert_close(actual,expected,rtol=1e-13,atol=1e-13)
    assert torch.equal(src, torch.arange(math.prod(shape),dtype=torch.float64).reshape(shape)*.2+10000)


def test_scatter_broadcast_different_groups_per_batch():
    src = torch.tensor([[[1.,2.],[3.,4.],[5.,6.]],[[7.,8.],[9.,10.],[11.,12.]]])
    index = torch.tensor([[2,0,2],[0,1,1]])
    summed = ops.scatter_sum(src,index,1)
    assert torch.equal(summed,torch.tensor([[[3.,4.],[0.,0.],[6.,8.]],[[7.,8.],[20.,22.],[0.,0.]]]))
    result = ops.scatter_softmax(src,index,1)
    assert torch.equal(result[0,1],torch.ones(2))
    torch.testing.assert_close(result[0,[0,2]].sum(0),torch.ones(2))
    torch.testing.assert_close(result[1,[1,2]].sum(0),torch.ones(2))


@pytest.mark.parametrize("function", [ops.scatter_sum,ops.scatter_softmax])
def test_scatter_empty_edges(function):
    result = function(torch.empty(2,0,3),torch.empty(0,dtype=torch.int64))
    assert tuple(result.shape)==(2,0,3)


def test_scatter_sum_integer_and_gradients():
    index = torch.tensor([2,0,2])
    assert torch.equal(ops.scatter_sum(torch.tensor([[1,3,5]]),index),torch.tensor([[3,0,6]]))
    src = torch.randn(1,3,2,dtype=torch.float64,requires_grad=True)
    assert torch.autograd.gradcheck(lambda x: ops.scatter_softmax(x,index), (src,))
    ops.scatter_sum(src,index).sum().backward()
    assert torch.equal(src.grad,torch.ones_like(src))


@pytest.mark.parametrize("call", [lambda:ops.neighbors(torch.tensor([1.]),torch.tensor([1])), lambda:ops.neighbors(torch.tensor([1]),torch.tensor([[1]])), lambda:ops.scatter_sum(torch.ones(1,2,3),torch.tensor([-1,0])), lambda:ops.scatter_sum(torch.ones(1,2,3),torch.tensor([0.,1.])), lambda:ops.scatter_sum(torch.ones(1,2,3),torch.tensor([0,1,2])), lambda:ops.scatter_softmax(torch.ones(1,2,dtype=torch.int64),torch.tensor([0,1]))])
def test_invalid_group_contracts(call):
    with pytest.raises(ValueError):
        call()


def pixel_oracle(net, b, c, y, x):
    if 0 <= y < net.shape[-2] and 0 <= x < net.shape[-1]:
        return float(net[b,c,y,x])
    return 0.


def fraction_fp32(value):
    # Native offsets are subtracted while coords still has FP32 dtype. Python
    # double subtraction would silently differ for small negative coordinates.
    return struct.unpack("f",struct.pack("f",value-math.floor(value)))[0]


def patch_oracle(net,coords,radius,bilinear=True):
    side = 2*radius+1 if bilinear else 2*radius+2
    out = torch.zeros(net.shape[0],coords.shape[1],net.shape[1],side,side,dtype=net.dtype)
    for b,m,c,oy,ox in itertools.product(range(net.shape[0]),range(coords.shape[1]),range(net.shape[1]),range(side),range(side)):
        x,y = map(float,coords[b,m])
        ix,iy = math.floor(x)+ox-radius, math.floor(y)+oy-radius
        if bilinear:
            dx,dy = fraction_fp32(x),fraction_fp32(y)
            out[b,m,c,oy,ox] = ((1-dy)*(1-dx)*pixel_oracle(net,b,c,iy,ix)+(1-dy)*dx*pixel_oracle(net,b,c,iy,ix+1)+dy*(1-dx)*pixel_oracle(net,b,c,iy+1,ix)+dy*dx*pixel_oracle(net,b,c,iy+1,ix+1))
        else:
            out[b,m,c,oy,ox] = pixel_oracle(net,b,c,iy,ix)
    return out


@pytest.mark.parametrize("radius,mode", [(0,"bilinear"),(1,"bilinear"),(1,"raw")])
def test_patchify_independent_xy_channel_patch_zero_padding_oracle(radius,mode):
    net = torch.arange(2*3*4*7,dtype=torch.float64).reshape(2,3,4,7)
    coords = torch.tensor([[[2.25,1.5],[-.25,.75],[6.5,3.25]],[[3.75,1.25],[-2.,2.],[7.25,-1.75]]],dtype=torch.float32)
    actual = ops.patchify(net,coords,radius,mode)
    expected = patch_oracle(net,coords,radius,mode=="bilinear")
    torch.testing.assert_close(actual,expected,rtol=1e-13,atol=1e-13)


def test_patchify_empty_and_half_fraction_dtype_and_noncontiguous_input():
    net = torch.arange(2*4*3,dtype=torch.float16).reshape(1,2,4,3).transpose(-1,-2)
    coords = torch.tensor([[[1.5,.5]]])
    out = ops.patchify(net,coords,0)
    assert out.dtype==torch.float32  # Official Python wrapper multiplies FP32 fractions.
    torch.testing.assert_close(out,patch_oracle(net.float(),coords,0))
    assert ops.patchify(net,torch.empty(1,0,2),1).shape==(1,0,2,3,3)


def corr_oracle(gmap,fmap,coords,ii,jj,radius):
    """Independent scalar CUDA-forward transcription; ox/y output axes explicit."""
    batch,_,channels,ph,pw = gmap.shape
    side = 2*radius+1
    out = torch.zeros(batch,len(ii),side,side,ph,pw,dtype=gmap.dtype)
    for b,e,ox,oy,py,px in itertools.product(range(batch),range(len(ii)),range(side),range(side),range(ph),range(pw)):
        x,y = map(float,coords[b,e,:,py,px])
        fx,fy = math.floor(x),math.floor(y)
        dx,dy = fraction_fp32(x),fraction_fp32(y)
        def dot(cy,cx):
            if not (0 <= cy < fmap.shape[-2] and 0 <= cx < fmap.shape[-1]):
                return 0.
            return sum(float(gmap[b,ii[e],c,py,px])*float(fmap[b,jj[e],c,cy,cx]) for c in range(channels))
        yy,xx = fy+oy-radius,fx+ox-radius
        out[b,e,ox,oy,py,px] = ((1-dx)*(1-dy)*dot(yy,xx)+dx*(1-dy)*dot(yy,xx+1)+(1-dx)*dy*dot(yy+1,xx)+dx*dy*dot(yy+1,xx+1))
    return out


def corr_fixture(edges=3):
    gen = torch.Generator().manual_seed(26)
    gmap = torch.randn(2,4,8,2,3,generator=gen,dtype=torch.float64)
    fmap = torch.randn(2,3,8,8,12,generator=gen,dtype=torch.float64)
    coords = torch.rand(2,edges,2,2,3,generator=gen)*torch.tensor([13.,9.])[None,None,:,None,None]-.7
    ii = torch.arange(edges,dtype=torch.int64)%4
    jj = (torch.arange(edges,dtype=torch.int64)+2)%3
    return gmap,fmap,coords,ii,jj


@pytest.mark.parametrize("radius", [0,1,3])
def test_corr_independent_loop_oracle_radius_xy_patch_channels_and_bounds(radius):
    args=corr_fixture()
    actual=ops.corr(*args,radius=radius)
    expected=corr_oracle(*args,radius)
    torch.testing.assert_close(actual,expected,rtol=1e-12,atol=1e-12)


def test_corr_65_edges_chunk_boundary_and_no_channel_normalization():
    gmap=torch.ones(1,2,8,1,1,dtype=torch.float64)
    fmap=torch.stack([torch.full((8,2,3),2.),torch.full((8,2,3),4.)]).double()[None]
    coords=torch.zeros(1,65,2,1,1)
    ii=torch.arange(65)%2
    jj=torch.arange(65)%2
    actual=ops.corr(gmap,fmap,coords,ii,jj,radius=0)
    assert torch.equal(actual.flatten(),torch.where(jj==0,16.,32.).double())


def pool_oracle(fmap,level):
    b,n,c,h,w=fmap.shape
    out=fmap.new_zeros(b,n,c,h//level,w//level)
    for bi,ni,ci,y,x in itertools.product(range(b),range(n),range(c),range(h//level),range(w//level)):
        out[bi,ni,ci,y,x] = sum(float(fmap[bi,ni,ci,y*level+yy,x*level+xx]) for yy in range(level) for xx in range(level))/(level*level)
    return out


def test_two_levels_independent_pool_and_level_innermost_flatten():
    gmap,fmap,coords,ii,jj=corr_fixture(2)
    fmap=fmap[...,:7,:11]  # pooling drops incomplete right/bottom cells.
    levels=(1,4)
    separate=[]
    for level in levels:
        pooled=pool_oracle(fmap,level)
        torch.testing.assert_close(ops.pyramidify(fmap,(level,))[0],pooled,rtol=1e-13,atol=1e-13)
        separate.append(corr_oracle(gmap,pooled,coords/level,ii,jj,1))
    actual=ops.corr_two_levels(gmap,fmap,coords,ii,jj,radius=1)
    expected=torch.stack(separate,-1).reshape(2,2,-1)
    torch.testing.assert_close(actual,expected,rtol=1e-12,atol=1e-12)
    torch.testing.assert_close(actual[...,0::2],separate[0].reshape(2,2,-1),rtol=1e-12,atol=1e-12)
    torch.testing.assert_close(actual[...,1::2],separate[1].reshape(2,2,-1),rtol=1e-12,atol=1e-12)


def test_corr_empty_and_forward_dropout_identity():
    args=corr_fixture(0)
    assert ops.corr(*args,radius=1).shape==(2,0,3,3,2,3)
    assert ops.corr_two_levels(*args,radius=1).shape==(2,0,108)
    args=corr_fixture(1)
    assert torch.equal(ops.corr(*args,dropout=0),ops.corr(*args,dropout=1))


@pytest.mark.parametrize("mutation", ["coordinate_dtype","nonfinite","channels_not_multiple8","patch_index","view_index","coordinate_shape","radius"])
def test_corr_invalid_native_contract(mutation):
    args=list(corr_fixture())
    kw={}
    if mutation=="coordinate_dtype": args[2]=args[2].double()
    elif mutation=="nonfinite": args[2][0,0,0,0,0]=float("nan")
    elif mutation=="channels_not_multiple8": args[0]=args[0][:,:,:7]; args[1]=args[1][:,:,:7]
    elif mutation=="patch_index": args[3][0]=4
    elif mutation=="view_index": args[4][0]=-1
    elif mutation=="coordinate_shape": args[2]=args[2][:,:,:,0]
    else: kw["radius"]=-1
    with pytest.raises(ValueError): ops.corr(*args,**kw)

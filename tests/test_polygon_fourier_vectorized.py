"""Boundary-integral vectorization against analytic and scalar references."""
import numpy as np
import pytest

from pcselsim.geometry import polygon_fourier_integral
from scripts.optimization.matched300_square_search import hole
from pcselsim.triangular_six_wave import TriangularPolygon


def scalar_reference(vertices,m,n):
    """Pre-optimization edge formula, with a sequential scalar accumulation."""
    points=np.asarray(vertices,float)
    area=.5*np.sum(points[:,0]*np.roll(points[:,1],-1)-np.roll(points[:,0],-1)*points[:,1])
    if area<0:points=points[::-1];area=-area
    if (m,n)==(0,0):return complex(area)
    wave=2*np.pi*np.array([m,n],float);value=0j
    for start,stop in zip(points,np.roll(points,-1,axis=0),strict=True):
        edge=stop-start;step=float(np.dot(wave,edge))
        average=1+0j if abs(step)<1e-13 else np.expm1(1j*step)/(1j*step)
        value+=np.dot(wave,np.array([edge[1],-edge[0]]))*np.exp(1j*np.dot(wave,start))*average
    return complex(value/(1j*np.dot(wave,wave)))


@pytest.mark.parametrize("angle",(0.,1e-12,23.,90.))
def test_rectangle_matches_analytic_transform_including_parallel_edges(angle):
    center=np.array([.173,-.217]);rx,ry=.19,.13
    theta=np.deg2rad(angle);rotation=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
    vertices=np.array([[-rx,-ry],[rx,-ry],[rx,ry],[-rx,ry]])@rotation.T+center
    for m,n in ((0,0),(0,7),(11,0),(3,-4),(71,89),(-173,129)):
        wave=2*np.pi*np.array([m,n]);local=rotation.T@wave
        expected=4*rx*ry*np.sinc(local[0]*rx/np.pi)*np.sinc(local[1]*ry/np.pi)*np.exp(1j*(wave@center))
        assert polygon_fourier_integral(tuple(map(tuple,vertices)),m,n)==pytest.approx(expected,rel=1e-12,abs=2e-16)


@pytest.mark.parametrize("kind",("superellipse","rounded_triangle","triangular_rounded"))
def test_vectorized_matches_preoptimization_formula_for_dense_high_order_contours(kind):
    if kind=="triangular_rounded":
        vertices=TriangularPolygon.rounded_regular((.127,-.219),.21,angle_deg=31.,
            corner_fraction=.3,samples_per_corner=32).vertices_fractional
    else:vertices=hole((.127,-.219),.075,kind,1.37,31.,3.7,96).vertices
    orders=[(m,n) for m in (-173,-48,-7,0,3,48,211) for n in (-129,-48,-1,0,11,48,197)]
    for contour in (vertices,vertices[::-1]):
        for m,n in orders:
            before=scalar_reference(contour,m,n);after=polygon_fourier_integral(contour,m,n)
            assert after==pytest.approx(before,rel=1e-12,abs=2e-16)


def test_translation_reciprocity_and_orientation_are_preserved():
    vertices=np.array(hole((0.,0.),.065,"rounded_triangle",1.29,37.,vertices=96).vertices)
    shift=np.array([.193,-.219])
    for m,n in ((0,0),(1,0),(7,-13),(48,47),(137,-199)):
        value=polygon_fourier_integral(tuple(map(tuple,vertices)),m,n)
        shifted=polygon_fourier_integral(tuple(map(tuple,vertices+shift)),m,n)
        assert shifted==pytest.approx(value*np.exp(2j*np.pi*(np.array([m,n])@shift)),rel=1e-12,abs=2e-16)
        assert polygon_fourier_integral(tuple(map(tuple,vertices[::-1])),m,n)==pytest.approx(value,abs=2e-16)
        assert polygon_fourier_integral(tuple(map(tuple,vertices)),-m,-n)==pytest.approx(value.conjugate(),abs=2e-16)

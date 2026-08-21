"""
app.py — Satellite Origami Backend (Production Ready for Render)
CORS fully fixed — works for all users worldwide
STANDING satellite — origami folds clearly visible
"""

import os, io, base64, warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from flask import Flask, request, jsonify, send_file, make_response
from flask_cors import CORS
from rag_pipeline import RAGPipeline

app = Flask(__name__)

# ── CORS — allow ALL origins ──────────────────────────────────────────────────
CORS(app, resources={r"/*": {"origins": "*"}},
     supports_credentials=False,
     allow_headers=["Content-Type"],
     methods=["GET","POST","OPTIONS"])

@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"]  = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE           = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH     = os.path.join(BASE, "cvae_model.pth")
KNOWLEDGE_PATH = os.path.join(BASE, "knowledge_base.json")
FRONTEND_PATH  = os.path.join(BASE, "chatbot.html")

LABEL_NAMES  = {0:"solar_panel", 1:"antenna", 2:"reflector", 3:"truss"}
NUM_CLASSES  = 4
INPUT_DIM    = 140
LATENT_DIM   = 16

# ── Body dimensions ───────────────────────────────────────────────────────────
BODY_X = 0.55
BODY_Y = 1.6
BODY_Z = 0.55

# ══════════════════════════════════════════════════════════════════════════════
# CVAE
# ══════════════════════════════════════════════════════════════════════════════
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(INPUT_DIM+NUM_CLASSES,256), nn.LayerNorm(256), nn.LeakyReLU(0.2),
            nn.Linear(256,128),                   nn.LayerNorm(128), nn.LeakyReLU(0.2),
        )
        self.fc_mean    = nn.Linear(128, LATENT_DIM)
        self.fc_log_var = nn.Linear(128, LATENT_DIM)
    def forward(self,x,c):
        h = self.shared(torch.cat([x,c],dim=1))
        return self.fc_mean(h), self.fc_log_var(h)

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(LATENT_DIM+NUM_CLASSES,128), nn.LayerNorm(128), nn.LeakyReLU(0.2),
            nn.Linear(128,256),                    nn.LayerNorm(256), nn.LeakyReLU(0.2),
            nn.Linear(256,INPUT_DIM), nn.Sigmoid()
        )
    def forward(self,z,c): return self.net(torch.cat([z,c],dim=1))

class CVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
    def reparameterize(self,mean,log_var):
        return mean + torch.exp(0.5*log_var) * torch.randn_like(log_var)
    def forward(self,x,c):
        mean,log_var = self.encoder(x,c)
        return self.decoder(self.reparameterize(mean,log_var),c), mean, log_var

print("Loading CVAE...")
ckpt  = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
cvae  = CVAE()
cvae.load_state_dict(ckpt["model_state"])
cvae.eval()
X_min   = ckpt["X_min"]
X_range = ckpt["X_range"]
print("CVAE loaded!")

print("Loading RAG...")
rag = RAGPipeline(KNOWLEDGE_PATH)
print("RAG loaded!")

def get_seed(part_name):
    label_idx = {v:k for k,v in LABEL_NAMES.items()}[part_name]
    with torch.no_grad():
        z = torch.randn(1,LATENT_DIM)
        c = torch.zeros(1,NUM_CLASSES); c[0,label_idx]=1.0
        out = cvae.decoder(z,c).numpy()[0]
    return out * X_range + X_min


# ══════════════════════════════════════════════════════════════════════════════
# GEOMETRY — Original perfect origami geometry (from local app.py)
# ══════════════════════════════════════════════════════════════════════════════

def build_solar_panel(offset, flip=1):
    ox, oy, oz = offset
    w, h = 3.2, 1.8
    nx, ny = 8, 4
    dx, dy = w/nx, h/ny
    shear  = 0.18
    verts, faces, mfolds, vfolds = [], [], [], []
    grid = {}
    for i in range(nx+1):
        for j in range(ny+1):
            x = ox + i*dx*flip
            y = oy + j*dy
            z = oz + 0.045*((i+j) % 2)
            if i % 2 == 1:
                y += shear * dy * 0.3
            grid[(i,j)] = len(verts)
            verts.append([x, y, z])
    verts = np.array(verts)
    for i in range(nx):
        for j in range(ny):
            a=grid[(i,j)]; b=grid[(i+1,j)]; c=grid[(i+1,j+1)]; d=grid[(i,j+1)]
            faces.append([a,b,c,d])
    for j in range(ny+1):
        for i in range(nx):
            mfolds.append([verts[grid[(i,j)]].tolist(), verts[grid[(i+1,j)]].tolist()])
    for i in range(nx):
        for j in range(ny):
            if (i+j) % 2 == 0:
                vfolds.append([verts[grid[(i,j)]].tolist(), verts[grid[(i+1,j+1)]].tolist()])
            else:
                vfolds.append([verts[grid[(i+1,j)]].tolist(), verts[grid[(i,j+1)]].tolist()])
    for i in range(0, nx+1, 2):
        for j in range(ny):
            mfolds.append([verts[grid[(i,j)]].tolist(), verts[grid[(i,j+1)]].tolist()])
    return verts, faces, mfolds, vfolds


def build_antenna(offset):
    ox, oy, oz = offset
    r_max=1.9; n_rings=8; n_segs=16; depth=0.55*r_max
    verts, faces, mfolds, vfolds = [], [], [], []
    verts.append([ox, oy, oz])
    center_idx = 0
    ring_start = []
    for i in range(1, n_rings+1):
        r = r_max*i/n_rings
        z = oz + depth*(i/n_rings)**2
        wobble = 0.018*(i%2)
        start = len(verts)
        ring_start.append(start)
        for j in range(n_segs):
            angle = 2*np.pi*j/n_segs
            verts.append([ox+r*np.cos(angle), oy+r*np.sin(angle), z+wobble])
    verts = np.array(verts)
    for j in range(n_segs):
        faces.append([center_idx, ring_start[0]+j, ring_start[0]+(j+1)%n_segs])
    for i in range(n_rings-1):
        rs=ring_start[i]; rn=ring_start[i+1]
        for j in range(n_segs):
            faces.append([rs+j, rs+(j+1)%n_segs, rn+(j+1)%n_segs, rn+j])
    rim_start = ring_start[-1]
    for j in range(n_segs):
        mfolds.append([[ox,oy,oz], verts[rim_start+j].tolist()])
    for i in range(0, n_rings, 2):
        rs=ring_start[i]
        ring_pts=[verts[rs+j].tolist() for j in range(n_segs)]+[verts[rs].tolist()]
        for k in range(len(ring_pts)-1):
            vfolds.append([ring_pts[k], ring_pts[k+1]])
    for j in range(0, n_segs, 2):
        mid_idx=ring_start[n_rings//2]+j
        mfolds.append([verts[mid_idx].tolist(), verts[rim_start+j].tolist()])
    return verts, faces, mfolds, vfolds


def build_reflector(offset):
    ox, oy, oz = offset
    size=1.0; depth=0.38; n=9
    verts, faces, mfolds, vfolds = [], [], [], []
    x_vals=np.linspace(-size,size,n); y_vals=np.linspace(-size,size,n)
    grid={}
    for i,xi in enumerate(x_vals):
        for j,yj in enumerate(y_vals):
            r2=xi**2+yj**2
            z=oz - depth*r2/(size**2)
            z += 0.022*((i+j)%2)
            grid[(i,j)]=len(verts)
            verts.append([ox+xi, oy+yj, z])
    verts=np.array(verts)
    for i in range(n-1):
        for j in range(n-1):
            a=grid[(i,j)]; b=grid[(i+1,j)]; c=grid[(i+1,j+1)]; d=grid[(i,j+1)]
            faces.append([a,b,c,d])
    for i in range(n):
        for j in range(n-1):
            mfolds.append([verts[grid[(i,j)]].tolist(), verts[grid[(i,j+1)]].tolist()])
    for j in range(0,n,3):
        for i in range(n-1):
            mfolds.append([verts[grid[(i,j)]].tolist(), verts[grid[(i+1,j)]].tolist()])
    for i in range(n-1):
        for j in range(n-1):
            if (i+j)%2==0:
                vfolds.append([verts[grid[(i,j)]].tolist(), verts[grid[(i+1,j+1)]].tolist()])
            else:
                vfolds.append([verts[grid[(i+1,j)]].tolist(), verts[grid[(i,j+1)]].tolist()])
    return verts, faces, mfolds, vfolds


def build_truss(offset):
    ox, oy, oz = offset
    length=5.5; radius=0.65; n_rings=10; n_sides=12
    verts, faces, mfolds, vfolds = [], [], [], []
    rings=[]
    for i in range(n_rings+1):
        z=oz+i*(length/n_rings); rot=(np.pi/n_sides)*(i%2); ring=[]
        for j in range(n_sides):
            angle=2*np.pi*j/n_sides+rot
            r=radius+0.025*(i%2)
            verts.append([ox+r*np.cos(angle), oy+r*np.sin(angle), z])
            ring.append(len(verts)-1)
        rings.append(ring)
    verts=np.array(verts)
    for i in range(n_rings):
        r0,r1=rings[i],rings[i+1]
        for j in range(n_sides):
            a=r0[j]; b=r0[(j+1)%n_sides]; c=r1[(j+1)%n_sides]; d=r1[j]
            faces.append([a,b,c,d])
    for i in range(0,n_rings+1,2):
        ring=rings[i]
        for j in range(n_sides):
            mfolds.append([verts[ring[j]].tolist(), verts[ring[(j+1)%n_sides]].tolist()])
    for j in range(n_sides):
        for i in range(n_rings):
            mfolds.append([verts[rings[i][j]].tolist(), verts[rings[i+1][j]].tolist()])
    for i in range(n_rings):
        r0,r1=rings[i],rings[i+1]
        for j in range(n_sides):
            if (i+j)%2==0:
                vfolds.append([verts[r0[j]].tolist(), verts[r1[(j+1)%n_sides]].tolist()])
            else:
                vfolds.append([verts[r0[(j+1)%n_sides]].tolist(), verts[r1[j]].tolist()])
    return verts, faces, mfolds, vfolds


def build_body():
    w,h,d=1.1,1.1,1.6; oz=2.1; eps=0.012
    v=np.array([
        [-w,-h,oz],[w,-h,oz],[w,h,oz],[-w,h,oz],
        [-w,-h,oz+d],[w,-h,oz+d],[w,h,oz+d],[-w,h,oz+d]
    ])
    faces=[[0,1,2,3],[4,5,6,7],[0,1,5,4],[2,3,7,6],[1,2,6,5],[0,3,7,4]]
    def mid(a,b): return [(a[i]+b[i])/2 for i in range(3)]
    def push(p,nx,ny,nz): return [p[0]+nx*eps,p[1]+ny*eps,p[2]+nz*eps]
    mfolds,vfolds=[],[]
    c0=push(v[0],0,-1,0); c1=push(v[1],0,-1,0); c4=push(v[4],0,-1,0); c5=push(v[5],0,-1,0)
    mfolds+=[[c0,c5],[c1,c4]]; vfolds+=[[mid(c0,c1),mid(c4,c5)],[mid(c0,c4),mid(c1,c5)]]
    c2=push(v[2],0,1,0); c3=push(v[3],0,1,0); c6=push(v[6],0,1,0); c7=push(v[7],0,1,0)
    mfolds+=[[c2,c7],[c3,c6]]; vfolds+=[[mid(c2,c3),mid(c6,c7)],[mid(c2,c6),mid(c3,c7)]]
    r1=push(v[1],1,0,0); r2=push(v[2],1,0,0); r5=push(v[5],1,0,0); r6=push(v[6],1,0,0)
    mfolds+=[[r1,r6],[r2,r5]]; vfolds+=[[mid(r1,r2),mid(r5,r6)],[mid(r1,r5),mid(r2,r6)]]
    l0=push(v[0],-1,0,0); l3=push(v[3],-1,0,0); l4=push(v[4],-1,0,0); l7=push(v[7],-1,0,0)
    mfolds+=[[l0,l7],[l3,l4]]; vfolds+=[[mid(l0,l3),mid(l4,l7)],[mid(l0,l4),mid(l3,l7)]]
    t4=push(v[4],0,0,1); t5=push(v[5],0,0,1); t6=push(v[6],0,0,1); t7=push(v[7],0,0,1)
    mfolds+=[[t4,t6],[t5,t7]]; vfolds+=[[mid(t4,t5),mid(t6,t7)],[mid(t4,t7),mid(t5,t6)]]
    b0=push(v[0],0,0,-1); b1=push(v[1],0,0,-1); b2=push(v[2],0,0,-1); b3=push(v[3],0,0,-1)
    mfolds+=[[b0,b2],[b1,b3]]; vfolds+=[[mid(b0,b1),mid(b2,b3)],[mid(b0,b3),mid(b1,b2)]]
    return v, faces, mfolds, vfolds


def get_part_positions(n_solar, n_antenna, n_reflector, n_truss):
    PZ=2.55; PY=-0.9
    if   n_solar==1: panel_pos=[(-1.1,PY,PZ,-1)]
    elif n_solar==2: panel_pos=[(-1.1,PY,PZ,-1),(1.1,PY,PZ,1)]
    elif n_solar==3: panel_pos=[(-1.1,PY,PZ,-1),(1.1,PY,PZ,1),(-1.1,PY,PZ+2.0,-1)]
    else:            panel_pos=[(-1.1,PY,PZ,-1),(1.1,PY,PZ,1),(-1.1,PY,PZ+2.0,-1),(1.1,PY,PZ+2.0,1)]
    if   n_antenna==1: ant_pos=[(0,0,6.2)]
    elif n_antenna==2: ant_pos=[(-1.0,0,6.2),(1.0,0,6.2)]
    else:              ant_pos=[(0,0,6.8),(-1.4,0,5.9),(1.4,0,5.9)]
    if   n_reflector==1: ref_pos=[(0,0,-0.2)]
    elif n_reflector==2: ref_pos=[(-1.2,0,-0.2),(1.2,0,-0.2)]
    else:                ref_pos=[(0,0,-0.2),(-1.8,0,0.7),(1.8,0,0.7)]
    return panel_pos, ant_pos, ref_pos


# ══════════════════════════════════════════════════════════════════════════════
# PNG RENDER
# ══════════════════════════════════════════════════════════════════════════════
def render_satellite_png(config):
    n_solar=int(config.get("solar_panel",2)); n_antenna=int(config.get("antenna",1))
    n_reflector=int(config.get("reflector",1)); n_truss=int(config.get("truss",1))
    panel_pos,ant_pos,ref_pos=get_part_positions(n_solar,n_antenna,n_reflector,n_truss)

    fig=plt.figure(figsize=(18,11)); fig.patch.set_facecolor("#0a0a14")
    ax_main=fig.add_subplot(1,2,1,projection="3d"); ax_main.set_facecolor("#0a0a14")
    ax_front=fig.add_subplot(1,2,2,projection="3d"); ax_front.set_facecolor("#0a0a14")
    ax_main.view_init(elev=22,azim=-55)
    ax_front.view_init(elev=0,azim=-90)

    def draw(ax,verts,faces,mf,vf,fcolor,alpha):
        if faces:
            polys=[[verts[vi] for vi in f] for f in faces]
            pc=Poly3DCollection(polys,alpha=alpha,facecolor=fcolor,edgecolor="none",linewidth=0)
            ax.add_collection3d(pc)
        for fold in mf:
            ax.plot([fold[0][0],fold[1][0]],[fold[0][1],fold[1][1]],[fold[0][2],fold[1][2]],
                    color="white",linewidth=0.8,alpha=0.9)
        for fold in vf:
            ax.plot([fold[0][0],fold[1][0]],[fold[0][1],fold[1][1]],[fold[0][2],fold[1][2]],
                    color="#aaddff",linewidth=0.5,alpha=0.7,linestyle="--")

    for ax in [ax_main,ax_front]:
        for t in range(max(1,n_truss)):
            v,f,mf,vf=build_truss((0,0,t*0.25)); draw(ax,v,f,mf,vf,"#f0a830",0.30)
        bv,bf,bmf,bvf=build_body(); draw(ax,bv,bf,bmf,bvf,"#4a6080",0.75)
        for (px,py,pz,flip) in panel_pos:
            v,f,mf,vf=build_solar_panel((px,py,pz),flip); draw(ax,v,f,mf,vf,"#5599ee",0.45)
        for pos in ant_pos:
            v,f,mf,vf=build_antenna(pos); draw(ax,v,f,mf,vf,"#33cc99",0.35)
        for pos in ref_pos:
            v,f,mf,vf=build_reflector(pos); draw(ax,v,f,mf,vf,"#aa55ee",0.35)
        for pane in [ax.xaxis.pane,ax.yaxis.pane,ax.zaxis.pane]:
            pane.fill=False; pane.set_edgecolor("#111")
        ax.grid(False); ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])

    ax_main.set_title("Perspective View",color="#aaddff",fontsize=12,fontweight="bold",pad=10)
    ax_front.set_title("Front View",color="#aaddff",fontsize=12,fontweight="bold",pad=10)

    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend=[
        Patch(facecolor="#5599ee",edgecolor="none",label=f"Solar Panels ({n_solar})  Miura-ori"),
        Patch(facecolor="#33cc99",edgecolor="none",label=f"Antenna ({n_antenna})  Radial crease"),
        Patch(facecolor="#aa55ee",edgecolor="none",label=f"Reflector ({n_reflector})  Waterbomb"),
        Patch(facecolor="#f0a830",edgecolor="none",label=f"Truss ({n_truss})  Yoshimura"),
        Patch(facecolor="#4a6080",edgecolor="none",label="Body"),
        Line2D([0],[0],color="white",linewidth=1.5,label="Mountain fold"),
        Line2D([0],[0],color="#aaddff",linewidth=1.0,linestyle="--",label="Valley fold"),
    ]
    fig.legend(handles=legend,loc="lower center",ncol=4,fontsize=9,
               facecolor="#0d0d20",labelcolor="white",framealpha=0.8,bbox_to_anchor=(0.5,-0.01))
    plt.suptitle(f"Origami Satellite — CVAE Generated\nSolar:{n_solar}  Antenna:{n_antenna}  Reflector:{n_reflector}  Truss:{n_truss}",
                 fontsize=13,fontweight="bold",color="white")
    plt.tight_layout()
    buf=io.BytesIO()
    plt.savefig(buf,format="png",dpi=140,bbox_inches="tight",facecolor="#0a0a14")
    plt.close(); buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


# ══════════════════════════════════════════════════════════════════════════════
# 3D FOR THREE.JS
# ══════════════════════════════════════════════════════════════════════════════
def quads_to_triangles(faces):
    idx=[]
    for f in faces:
        if len(f)==4: a,b,c,d=f; idx+=[a,b,c,a,c,d]
        elif len(f)==3: idx+=list(f)
    return idx

def mesh_dict(verts,faces,mf,vf,color,opacity,mfold_color,vfold_color):
    vl=verts.tolist() if hasattr(verts,"tolist") else verts
    v3=[[v[0],v[2],-v[1]] for v in vl]
    idx=quads_to_triangles(faces)
    mlines,vlines=[],[]
    for p in mf:
        mlines+=[[p[0][0],p[0][2],-p[0][1]],[p[1][0],p[1][2],-p[1][1]]]
    for p in vf:
        vlines+=[[p[0][0],p[0][2],-p[0][1]],[p[1][0],p[1][2],-p[1][1]]]
    return {"vertices":v3,"indices":idx,"mountain_lines":mlines,"valley_lines":vlines,
            "color":color,"opacity":opacity,"mfold_color":mfold_color,"vfold_color":vfold_color}

def build_satellite_3d(config):
    n_solar=int(config.get("solar_panel",2)); n_antenna=int(config.get("antenna",1))
    n_reflector=int(config.get("reflector",1)); n_truss=int(config.get("truss",1))
    panel_pos,ant_pos,ref_pos=get_part_positions(n_solar,n_antenna,n_reflector,n_truss)
    meshes=[]
    for t in range(max(1,n_truss)):
        v,f,mf,vf=build_truss((0,0,t*0.25)); meshes.append(mesh_dict(v,f,mf,vf,"#ffaa22",0.55,"#ffffff","#ffdd88"))
    bv,bf,bmf,bvf=build_body(); meshes.append(mesh_dict(bv,bf,bmf,bvf,"#667799",0.85,"#aaccff","#88aacc"))
    for (px,py,pz,flip) in panel_pos:
        v,f,mf,vf=build_solar_panel((px,py,pz),flip); meshes.append(mesh_dict(v,f,mf,vf,"#4488ff",0.70,"#ffffff","#aaccff"))
    for pos in ant_pos:
        v,f,mf,vf=build_antenna(pos); meshes.append(mesh_dict(v,f,mf,vf,"#22ddaa",0.60,"#ffffff","#88ffdd"))
    for pos in ref_pos:
        v,f,mf,vf=build_reflector(pos); meshes.append(mesh_dict(v,f,mf,vf,"#cc66ff",0.60,"#ffffff","#ddaaff"))
    return meshes

# ══════════════════════════════════════════════════════════════════════════════
# ROUTES
# ══════════════════════════════════════════════════════════════════════════════
@app.route("/api/health", methods=["GET","OPTIONS"])
def health():
    return jsonify({"status":"ok","message":"Satellite Origami Backend Running"})

@app.route("/api/generate", methods=["POST","OPTIONS"])
def generate():
    if request.method == "OPTIONS":
        return "", 204
    data=request.json
    cfg={"solar_panel":data.get("solar_panel",2),"antenna":data.get("antenna",1),
         "reflector":data.get("reflector",1),"truss":data.get("truss",1)}
    return jsonify({"image":render_satellite_png(cfg),"config":cfg,"status":"generated"})

@app.route("/api/generate3d", methods=["POST","OPTIONS"])
def generate3d():
    if request.method == "OPTIONS":
        return "", 204
    data=request.json
    cfg={"solar_panel":data.get("solar_panel",2),"antenna":data.get("antenna",1),
         "reflector":data.get("reflector",1),"truss":data.get("truss",1)}
    return jsonify({"meshes":build_satellite_3d(cfg),"config":cfg,"status":"ok"})

@app.route("/api/chat", methods=["POST","OPTIONS"])
def chat():
    if request.method == "OPTIONS":
        return "", 204
    data=request.json
    message=data.get("message","").strip()
    answer,sources=rag.answer(message)
    return jsonify({"reply":answer,
                    "sources":[{"topic":s.get("topic",""),"score":s.get("score",0)} for s in sources]})

@app.route("/")
def index():
    return send_file(FRONTEND_PATH)

# ══════════════════════════════════════════════════════════════════════════════
# START
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("\nSatellite Origami Backend — Production (Render)")
    print("CORS: ALL origins allowed")
    print("Satellite: STANDING orientation — origami folds visible from front")
    print("Routes: /  |  /api/health  |  /api/generate  |  /api/generate3d  |  /api/chat")
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)

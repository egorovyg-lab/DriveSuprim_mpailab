"""Input-only top4 camera overlays; OpenScene camera extrinsics, flat ego z=0."""
import io
import hashlib
import numpy as np

import real_data as base

MODES=('overlay','separate','blank')
ORDERS=('canonical','reversed','permuted')
TASKS=tuple((m,o) for m in MODES for o in ORDERS)
CONFIG=dict(protocol='overlay_top4_v1',candidates=4,camera='cam_f0',image_size=[768,432],
    camera_frame='OpenScene raw nuPlan camera-to-ego stored as sensor2lidar',ground_z=0.,
    distortion='OpenCV Brown-Conrady 0/4/5/8 coefficients',tasks=TASKS,
    primary='overlay_mean3',max_image_pixels=2048*1024,max_length=8192,images_per_request=5)
NOTE=('Exploratory train/development replay. Perspective overlays assume a flat z=0 road in '
      'the current ego frame; not a reconstruction of slopes, occlusions or future road height. '
      'Only current front camera, fixed top4 subset of the original top16. No route command, '
      'new labels, fitting or fresh holdout. Compare oracle_top4 separately from oracle_top16. '
      'Stable candidate IDs, three orders per condition. Separate is a representation comparison; '
      'overlay vs blank is the matched road-photograph control. Raw local PDM, not full EPDMS.')


def calibration(scene,size):
    c=scene['current_camera_calibrations']['cam_f0']
    r=np.asarray(c['sensor2lidar_rotation'],dtype=float)
    t=np.asarray(c['sensor2lidar_translation'],dtype=float).reshape(-1)
    k=np.asarray(c['cam_intrinsic'],dtype=float)
    d=np.asarray(c.get('distortion',[]),dtype=float).reshape(-1)
    if (r.shape!=(3,3) or t.shape!=(3,) or k.shape!=(3,3) or len(d) not in (0,4,5,8)
            or not all(np.isfinite(a).all() for a in (r,t,k,d))):
        raise ValueError('Invalid/unsupported camera calibration; no fallback to guessed extrinsics')
    if not np.allclose(r.T@r,np.eye(3),atol=1e-4) or not np.isclose(np.linalg.det(r),1,atol=1e-4):
        raise ValueError('Camera rotation is not a proper SO(3) matrix')
    w,h=size
    if (k[0,0]<=0 or k[1,1]<=0 or not 0<k[0,2]<w or not 0<k[1,2]<h
            or not np.allclose(k[2],[0,0,1]) or abs(k[0,1])>1e-6 or abs(k[1,0])>1e-6):
        raise ValueError('Intrinsics/image dimensions inconsistent')
    if not .2<t[2]<4 or r[0,2]<.5 or r[2,1]>-.5:
        raise ValueError('Front camera axes/height inconsistent with current ego frame')
    return r,t,k,d


def project(points,calib):
    r,t,k,d=calib
    # Camera-to-ego p_ego=R p_camera+t -> p_camera=R^T(p_ego-t).
    camera=(np.asarray(points,dtype=float)-t)@r
    depth=camera[:,2]
    valid=depth>.2
    normalized=np.zeros((len(camera),2)); normalized[valid]=camera[valid,:2]/depth[valid,None]
    x,y=normalized.T; rr=x*x+y*y
    coeff=np.zeros(8); coeff[:len(d)]=d
    k1,k2,p1,p2,k3,k4,k5,k6=coeff
    with np.errstate(over='ignore',divide='ignore',invalid='ignore'):
        denominator=1+k4*rr+k5*rr**2+k6*rr**3
        radial=(1+k1*rr+k2*rr**2+k3*rr**3)/denominator
        xd=x*radial+2*p1*x*y+p2*(rr+2*x*x)
        yd=y*radial+p1*(rr+2*y*y)+2*p2*x*y
        uv=np.column_stack([k[0,0]*xd+k[0,2],k[1,1]*yd+k[1,2]])
    valid &= np.isfinite(uv).all(axis=1)&(np.abs(denominator)>1e-8)
    return uv,valid


def clipped_segment(a,b,w,h):
    lo,hi=0.,1.; delta=b-a
    for dim,limit in ((0,w-1),(1,h-1)):
        if abs(delta[dim])<1e-12:
            if not 0<=a[dim]<=limit: return None
        else:
            v0,v1=-a[dim]/delta[dim],(limit-a[dim])/delta[dim]
            lo=max(lo,min(v0,v1)); hi=min(hi,max(v0,v1))
            if lo>hi: return None
    return [tuple(a+lo*delta),tuple(a+hi*delta)]


def path_segments(poses,calib,size):
    xy=np.vstack([np.zeros((1,2)),poses[:,:2]])
    dense=np.concatenate([a[None,:]+np.linspace(0,1,9)[:,None]*(b-a)[None,:] for a,b in zip(xy,xy[1:])])
    uv,valid=project(np.column_stack([dense,np.zeros(len(dense))]),calib)
    segments=[]
    for i in range(len(uv)-1):
        if valid[i] and valid[i+1]:
            line=clipped_segment(uv[i],uv[i+1],*size)
            if line is not None: segments.append(line)
    return segments


def ids(token):
    # Random stable names independent of teacher rank, unchanged by image/list permutations.
    ranked=sorted(range(4),key=lambda i:base.digest([base.SEED,token,'names',i]))
    return {pid:'T'+chr(65+j) for j,pid in enumerate(ranked)}


def order(token,kind):
    canonical=sorted(range(4),key=lambda i:base.digest([base.SEED,token,'order',i]))
    if kind=='canonical': return canonical
    if kind=='reversed': return canonical[::-1]
    if kind=='permuted': return canonical[1:]+canonical[:1]
    raise ValueError('Unknown order')


def inset(trajectories,pid):
    from PIL import Image,ImageDraw
    im=Image.new('RGB',(160,120),'white'); draw=ImageDraw.Draw(im)
    xy=np.concatenate([np.zeros((1,2)),trajectories[:4,:,:2].reshape(-1,2)])
    lo=xy.min(axis=0)-1; hi=xy.max(axis=0)+1
    scale=min(95/(hi[0]-lo[0]),140/(hi[1]-lo[1])); middle=(hi+lo)/2
    pixel=lambda p:(80-(p[1]-middle[1])*scale,55-(p[0]-middle[0])*scale)
    points=[pixel((0,0))]+[pixel(p) for p in trajectories[pid,:,:2]]
    draw.line(points,fill='#0066dd',width=2)
    px,py=points[0]; draw.ellipse((px-2,py-2,px+2,py+2),fill='black')
    draw.text((3,104),'Full path: up=forward',fill='black',font=base.font(10))
    draw.rectangle((0,0,159,119),outline='black')
    return im


def pictures(photo,trajectories,calib,names,mode,ordering):
    from PIL import Image,ImageDraw,ImageOps
    size=tuple(CONFIG['image_size'])
    reference=ImageOps.pad(photo,size,color='white') if mode!='blank' else Image.new('RGB',size,'white')
    output=[reference]
    for pid in ordering:
        original=photo.copy() if mode=='overlay' else Image.new('RGB',photo.size,'white')
        draw=ImageDraw.Draw(original)
        if mode in ('overlay','blank'):
            for line in path_segments(trajectories[pid],calib,photo.size):
                draw.line(line,fill='#00bfff',width=max(2,round(photo.width/450)))
        elif mode!='separate': raise ValueError('Unknown representation mode')
        rendered=ImageOps.pad(original,size,color='white')
        draw=ImageDraw.Draw(rendered)
        # Identical text/inset in all conditions. Same untouched reference photograph.
        draw.rectangle((0,0,size[0],28),fill='white')
        draw.text((8,4),f'{names[pid]} | 4 second candidate | flat-ground approximation',fill='black',font=base.font(16))
        rendered.paste(inset(trajectories,pid),(size[0]-168,size[1]-128))
        if mode=='separate':
            # A larger separate BEV schematic, using the same geometry and scale convention.
            big=inset(trajectories,pid).resize((480,360))
            rendered.paste(big,(120,40))
        output.append(rendered)
    return output


def payload(package,scene,manifest,sample,mode,kind):
    from PIL import Image
    token=scene['metadata']['token']
    path=base.camera_paths(package,scene,manifest)[4]  # current front, [previous L,F,R,current L,F,R]
    with Image.open(path) as f: photo=f.convert('RGB')
    calib=calibration(scene,photo.size)
    names=ids(token); ordering=order(token,kind)
    images=pictures(photo,sample['trajectories'],calib,names,mode,ordering)
    row=dict(state=dict(task='Select the best four-second ego trajectory from four candidates.',
        image_1='Unannotated current front view; may be blank in the road-image control.',
        candidate_images='Images 2..5 are the four candidates, each labelled with its stable ID. '
            'Blue projected centrelines assume flat ground and do not model occlusion. '
            'Small insets show the entire path, up=forward and left=left, same metric scale. '
            'Projected parts behind the camera or outside its view are not drawn. '
            'Separate-schematic condition shows the path without a road background. '
            'Missing projected pixels do not mean the vehicle stops.',
        velocity_xy_mps=sample['ego_velocity'],acceleration_xy_mps2=sample['ego_acceleration']),
        question=dict(type='choice',criteria={names[pid]:f'Candidate {names[pid]} in image {j+2}.' for j,pid in enumerate(ordering)},
            instructions='Choose the best available candidate. Prioritize staying on the drivable road, '
            'avoiding collisions and obeying traffic rules, then useful progress and comfort. '
            'Read the full path inset as well as visible projected parts. No candidate is labelled as baseline.'),images=images)
    hashes=[]
    for image in images:
        b=io.BytesIO(); image.save(b,format='PNG'); hashes.append(hashlib.sha256(b.getvalue()).hexdigest())
    h=base.digest(dict(row={k:v for k,v in row.items() if k!='images'},images=hashes,pool_positions=ordering))
    return row,ordering,h

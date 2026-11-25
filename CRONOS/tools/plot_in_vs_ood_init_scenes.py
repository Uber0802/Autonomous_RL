"""In-domain vs out-of-domain start-scene figure (run from CRONOS/).

In-domain frames: full-res (640x480) default-scene renders from `eval_only.py --render-init-only
--start-scene default --seed 0` (seq0/task0), copied to figures/in_domain_init_scenes/.
Out-of-domain frames: figures/Out_of_Domain_Configuration/ only. Columns pair each OOD
image with an in-domain scene from the same object group."""
import matplotlib.pyplot as plt, matplotlib.image as mpimg
R='figures/'; ID=R+'in_domain_init_scenes/in_domain_group_'; OOD=R+'Out_of_Domain_Configuration/'
cols=[('B_seq0_task0_env0','overlap_2_gpu2_env109','Overlap'),('C_seq0_task0_env0','overlap_3_gpu4_env17','Overlap'),
      ('D_seq0_task0_env0','overlap_4_gpu6_env16','Overlap'),('A_seq0_task0_env0','drop_5_gpu1_env83','Drop'),
      ('C_seq0_task0_env3','drop_3_gpu4_env66','Drop'),('D_seq0_task0_env1','drop_4_gpu6_env0','Drop')]
plt.rcParams.update({'font.family':'serif','font.size':13})
fig,ax=plt.subplots(2,6,figsize=(18,4.4),gridspec_kw=dict(wspace=0.03,hspace=0.05))
for j,(i,o,kind) in enumerate(cols):
    ax[0,j].imshow(mpimg.imread(ID+i+'.png')); ax[1,j].imshow(mpimg.imread(OOD+o+'.png'))
    for a in ax[:,j]: a.set_xticks([]); a.set_yticks([]); [s.set_visible(False) for s in a.spines.values()]
ax[0,0].set_ylabel('In-Domain',fontsize=14,fontweight='bold'); ax[1,0].set_ylabel('Out-of-Domain',fontsize=14,fontweight='bold')
for ext in ('png','pdf'): fig.savefig(R+f'in_vs_out_of_domain_init_scenes.{ext}',dpi=200,bbox_inches='tight',pad_inches=0.02)

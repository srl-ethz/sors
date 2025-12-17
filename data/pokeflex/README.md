# Data for Pokeflex Poking Example

We store the reconstructed pokeflex data in: https://www.dropbox.com/scl/fo/mdud9w2mh0d609jv8oxlr/APxSJwiUX1ysTkreCXPfyZo?rlkey=j2sbn7ts4v41dhfkg7vyh4lo7&st=pu5shkjs&dl=0

- FoamDice_T1 to FoamDice_T8: contains all reconstructed data of all pokes during all trials. Each trial T has the mesh data, mesh confidence data and robot data.
- intitialMeshes: contains the initial meshes of 10 selected pokes of foam dice as .obj file. The data is annotated with T for trial number and P for poke number. We use these to get the .msh files in meshes/initialMeshesPokingFoamDice.

To run demos d03a_pokeflex_sim2real.py and d03b_pokeflex_opt.py, download the contents of this folder and place here.
#!/usr/bin/env python3
"""Prepare a separate PhysX import URDF without modifying the supplied model."""
from pathlib import Path
import xml.etree.ElementTree as ET
import json
ROOT=Path(__file__).resolve().parents[1]
src=ROOT/'models/source/human_39dof/human_description/urdf/human.urdf'
tree=ET.parse(src);robot=tree.getroot();modified=[]
for node in list(robot):
    if (node.tag=='link' and node.get('name')=='base_link') or (node.tag=='joint' and node.get('name')=='base_to_pelvis'):robot.remove(node)
for mesh in robot.findall('.//mesh'):
    mesh.set('filename','../../source/human_39dof/human_description/meshes/'+mesh.get('filename').split('meshes/')[-1])
for link in robot.findall('link'):
    if link.find('inertial') is None:
        i=ET.SubElement(link,'inertial');ET.SubElement(i,'origin',xyz='0 0 0',rpy='0 0 0');ET.SubElement(i,'mass',value='0.0001');ET.SubElement(i,'inertia',ixx='0.000001',iyy='0.000001',izz='0.000001',ixy='0',ixz='0',iyz='0');modified.append(link.get('name'))
for j in robot.findall('joint'):
    if j.get('type')=='revolute':
        # Exported velocity limits are placeholder hardware values; motion is up to 6 rad/s.
        j.find('limit').set('velocity','20')
        dynamics=j.find('dynamics')
        if dynamics is not None:dynamics.set('damping','0')
out=ROOT/'models/processed/human39';out.mkdir(parents=True,exist_ok=True);ET.indent(tree);tree.write(out/'human_training.urdf',encoding='utf-8',xml_declaration=True)
(ROOT/'reports/training_model_changes.json').write_text(json.dumps({'source_unchanged':True,'root':'PelvisLink (remove ground-frame fixed offset)','tiny_mass_links':modified,'added_mass_kg':len(modified)*.0001,'velocity_limit_rad_s':20,'joint_passive_damping':0,'reason':'explicit PD computes damping; placeholder velocity limits below source motion; regularize massless serial joint frames for PhysX'},indent=2))

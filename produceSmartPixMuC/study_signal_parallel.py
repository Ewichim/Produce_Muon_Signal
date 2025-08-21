import pyLCIO
import ROOT
import glob
import os
import numpy as np
from utils import sensorAngles, getYlocalAndGamma
from plothelper import *

ROOT.gROOT.SetBatch()
plt = PlotHelper()
plot = False
max_events = -1
max_npart = 100000
Bfield = 3.57
npart = 0
nevts = 0
tracks = []

# Gather all output_sim_*.slcio files
slcio_files = sorted(glob.glob("output_sim_*.slcio"))
print(f"Found {len(slcio_files)} files: {slcio_files}")

for file_path in slcio_files:
    print(f"Processing {file_path}")
    reader = pyLCIO.IOIMPL.LCFactory.getInstance().createLCReader()
    reader.open(file_path)
    for ievt, event in enumerate(reader):
        nevts += 1
        if max_events != -1 and nevts > max_events:
            break
        collection_names = event.getCollectionNames()
        vtxBarrelHits = event.getCollection("VertexBarrelCollection")
        hit_id_last = -1
        for hit in vtxBarrelHits:
            position = hit.getPosition()
            x, y, hit_z = position[0], position[1], position[2]
            rxy = (x ** 2 + y ** 2) ** 0.5
            t = hit.getTime()
            encoding = vtxBarrelHits.getParameters().getStringVal(pyLCIO.EVENT.LCIO.CellIDEncoding)
            decoder = pyLCIO.UTIL.BitField64(encoding)
            cellID = int(hit.getCellID0())
            decoder.setValue(cellID)
            detector = decoder["system"].value()
            layer = decoder['layer'].value()
            side = decoder["side"].value()
            if layer != 0:
                continue
            mcp = hit.getMCParticle()
            hit_pdg = mcp.getPDG() if mcp else None
            hit_id = mcp.id() if mcp else None
            if abs(hit_pdg) != 13 or abs(hit_pdg) != 11:
                continue
            mcp_p = mcp.getMomentum()
            mcp_tlv = ROOT.TLorentzVector()
            mcp_tlv.SetPxPyPzE(mcp_p[0], mcp_p[1], mcp_p[2], mcp.getEnergy())
            hit_p = hit.getMomentum()
            hit_tlv = ROOT.TLorentzVector()
            hit_tlv.SetPxPyPzE(hit_p[0], hit_p[1], hit_p[2], mcp.getEnergy())
            prodx, prody, prodz = mcp.getVertex()[0], mcp.getVertex()[1], mcp.getVertex()[2]
            endx, endy, endz = mcp.getEndpoint()[0], mcp.getEndpoint()[1], mcp.getEndpoint()[2]
            prodrxy = (prodx ** 2 + prody ** 2) ** 0.5
            endrxy = (endx ** 2 + endy ** 2) ** 0.5
            ylocal, gamma0 = getYlocalAndGamma(x, y)
            zglobal = round(hit_z / 25e-3) * 25e-3
            theta = hit_tlv.Theta()
            phi = hit_tlv.Phi()
            xvec = np.sin(theta) * np.cos(phi)
            yvec = np.sin(theta) * np.sin(phi)
            zvec = np.cos(theta)
            yp = xvec * np.sin(np.pi / 2 - gamma0) + yvec * np.cos(np.pi / 2 - gamma0)
            alpha = np.arctan2(yp, zvec)
            beta = phi - (gamma0 - np.pi / 2)
            cotb = 1. / np.tan(beta + np.pi)
            cota = 1. / np.tan(2 * np.pi - alpha)
            ylocal *= -1
            p = mcp_tlv.P()
            pt = mcp_tlv.Pt()
            track = [cota, cotb, p, 0, ylocal, zglobal, pt, t, hit_pdg]
            tracks.append(track)
            hit_id_last = hit_id
    reader.close()

# Save tracklists as before
file_path = "./signal_tracklists"
float_precision = 5
binsize = 500
numFiles = int(np.ceil(len(tracks) / binsize))
append = False
startNum = 0
if not os.path.isdir(file_path):
    os.makedirs(file_path)
else:
    if not append:
        files = os.listdir(file_path)
        for f in files:
            if os.path.isfile(f"{file_path}/{f}"):
                os.remove(f"{file_path}/{f}")
    else:
        files = glob.glob(file_path + "/*.txt")
        startNum = len(files)
for fileNum in range(startNum, startNum + numFiles):
    with open(f"{file_path}/sig_tracklist{fileNum}.txt", 'w') as file:
        for track in tracks[fileNum * binsize:(fileNum + 1) * binsize]:
            track = list(track)
            formatted_sublist = [f"{element:.{float_precision}f}" if isinstance(element, float) else element for element in track]
            line = ' '.join(map(str, formatted_sublist)) + '\n'
            file.write(line)
print(f"Saved {numFiles} tracklist files to {file_path}") 
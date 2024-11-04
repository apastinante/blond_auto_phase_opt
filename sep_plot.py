import numpy as np
from matplotlib import pyplot as plt
from functools import partial
from skimage import measure

def separatrix2rf_computer(eta, beta, energy, charge, voltage, harmonic1, vratio, hratio, phi_s1, phi_s2 , H_outer):
    """
    Compute the separatrix in the Double RF system 
    NOTE: Only works for non-accelerating system, otherwise it would involve a numerical solver
    """
    
    # IMPLEMENT TWO SEPARATRIX ONE INNER AND ONE OUTER 
    eom_factor_potential = -np.sign(eta)
    phase_array = np.linspace(-np.pi, np.pi, 1000,endpoint=True)

    # Compute phi_b 


    v_s = np.sqrt(harmonic1*charge*voltage*np.abs(eta)/(2*np.pi*beta**2 * energy))
    
    
 
        
    separatrix_array_outer = np.sqrt((H_outer + eom_factor_potential*v_s*(np.cos(phi_s1*np.pi/180) -np.cos(phase_array) + (phi_s1*np.pi/180 - phase_array)*np.sin(phi_s1*np.pi/180) - \
                vratio/hratio *(np.cos(phi_s2*np.pi/180) - np.cos(phi_s2*np.pi/180 + hratio*(phase_array -phi_s1*np.pi/180)) \
                    - hratio*(phase_array - phi_s1*np.pi/180)*np.sin(phi_s2*np.pi/180))) )*2*beta**2 * energy/ (harmonic1*eta))

    sep_array_o = separatrix_array_outer
    sep_array_o = np.append(separatrix_array_outer, -separatrix_array_outer[::-1])
    phase_sep = np.append(phase_array, phase_array[::-1])  

    return sep_array_o[np.isfinite(sep_array_o)]
    
    
        

    


def hamiltonian2rf(eta, beta, energy, charge, voltage, harmonic1, vratio, hratio, phi_s1, phi_s2 , phi,de):
    """Compute the hamiltonian in the Double RF system"""
    eom_factor_potential = -np.sign(eta)
    v_s = np.sqrt(harmonic1*charge*voltage*np.abs(eta)/(2*np.pi*beta**2 * energy))
    V = v_s *((np.cos(phi_s1) - np.cos(phi)) + (phi_s1 - phi)*np.sin(phi_s1) - vratio/hratio *(np.cos(phi_s2) - np.cos(phi_s2 + hratio*(phi - phi_s1)) - hratio*(phi - phi_s1)*np.sin(phi_s2)))
    H = -eom_factor_potential*(v_s*0.5*(de/(beta**2 * energy)*np.abs(eta)/v_s)**2 + V)

    return H


def closedness(contour):
    # Check if the contour is closed
    cond1 = contour[0][0] == contour[-1][0]
    cond2 = contour[0][1] == contour[-1][1]
    
    if np.any(cond1) and np.any(cond2):
           return True
    else:
        return False

def compute_outer_sep(Phi,De,Hs, dH):
    # Numerically computes the outer separatrix in the Double RF system
    # Phi, De are np.meshgrid
    # Hs is the hamiltonian value at the meshgrid
    # Find the outer separatrix by findind the contour with the highest H value where the separatrix is not given by an empty array
    H_contour = dH
    condition = True
    double_hit = False
    while condition: 
        contour = measure.find_contours(Hs, H_contour)
        if len(contour) == 2:
            double_hit = True
        if double_hit:
            count = len(contour)
            if count == 1:
                condition = False
        exit = closedness(contour[0])
        if not exit:
            condition = False
        else:
            H_contour += dH
    return H_contour



    
energy=10     #typically [MeV], [GeV], [TeV]
beta=0.9      #relativistic velocity factor
charge=1      #in units of the electron charge, just unity for protons
voltage=0.1   #RF voltage, typically in units of [V], [kV], [MV] 
harmonic=1    #Harmonic number of the RF system
eta=0.01      #1/gamma**2-1/gamma_transition**2
vratio = 0.5
hratio = 2
phis1 = np.pi/6 
phis2 = np.pi 

dH = 0.000001
# Create a mesh of phi and delta E points

phi= np.linspace(-np.pi-0.1, np.pi+0.1, 1000,endpoint=False)
de = np.linspace(-8, 8, 1000,endpoint=False)

ham_filled = partial(separatrix2rf_computer, eta, beta, energy, charge, voltage, harmonic, vratio, hratio, phis1, phis2)

H_out= hamiltonian2rf(eta, beta, energy, charge, voltage, harmonic, 0.6, 2, 0, 0, np.pi, 0)


Phi, De = np.meshgrid(phi, de)



Hs = hamiltonian2rf(eta, beta, energy, charge, voltage, harmonic, vratio, hratio, phis1, phis2, Phi, De)
h_outer  =  compute_outer_sep(Phi,De,Hs, dH)
# Extract the x and y indexes of the contour by rounding the x and y values
# x_contour = np.round(contour[:, 0]).astype(int)
# y_contour = np.round(contour[:, 1]).astype(int)
# # Extract the values of the contour
# phi_outer = Phi[x_contour, y_contour]
# de_outer = De[x_contour, y_contour]

fig, ax = plt.subplots()
ax.contour(Phi, De, Hs, levels=list(np.linspace(0, h_outer*1.5, 20))[1:])

ax.contour(Phi, De, Hs, levels=[0], colors='r')
ax.contour(Phi, De, Hs, levels=[h_outer-dH], colors='k')
ax.set_xlabel(r"$\phi$ [rad]")
ax.set_ylabel(r"$\Delta E$ [eV]")
ax.set_title(r'Hamiltonian Contour Plot for $\Phi_2 = {y}$'.format(y=round(phis2, ndigits=3)))
# ax.set_title(r'Hamiltonian Contour Plot for $\phi_{s0} = $' + str(round(phis1, ndigits=3)) + ' and $\phi_s = $' + '{y}'.format(y=round(phis2, ndigits=3)))
plt.show()




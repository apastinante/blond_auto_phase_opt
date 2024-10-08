
import matplotlib.pyplot as plt
import numpy as np
import os 
from copy import deepcopy

def multiple_formatter(denominator=2, number=np.pi, latex=r'\pi'):
    def gcd(a, b):
        while b:
            a, b = b, a%b
        return a
    def _multiple_formatter(x, pos):
        den = denominator
        num = np.int(np.rint(den*x/number))
        com = gcd(num,den)
        (num,den) = (int(num/com),int(den/com))
        if den==1:
            if num==0:
                return r'$0$'
            if num==1:
                return r'$%s$'%latex
            elif num==-1:
                return r'$-%s$'%latex
            else:
                return r'$%s%s$'%(num,latex)
        else:
            if num==1:
                return r'$\frac{%s}{%s}$'%(latex,den)
            elif num==-1:
                return r'$\frac{-%s}{%s}$'%(latex,den)
            else:
                return r'$\frac{%s%s}{%s}$'%(num,latex,den)
    return _multiple_formatter

class Multiple:
    def __init__(self, denominator=2, number=np.pi, latex= r'\pi'):
        self.denominator = denominator
        self.number = number
        self.latex = latex
    def locator(self):
        return plt.MultipleLocator(self.number / self.denominator)
    def formatter(self):
        return plt.FuncFormatter(multiple_formatter(self.denominator, self.number, self.latex))

class ProgramDef(object):
    def __init__(self,v_1,v_2,phase,time=None, sync_momentum = None, t_range = None, windowed = False, denominator_phase = 4):
        """
        This is a class to plot the voltage and phase over time and 
        iteratively updates them depending on the user clicks on the figure.

        Intructions: 
        Whatever point is clicked, it will update the value of that point
        at that given time and all subsequent points.
        The program will be closed when the user closes a figure.

        To set the voltage to 0, it is sufficient to click a value below the 
        x axis. This will set all subsequent values to 0.


        Parameters
        ----------
        v_1 : array
            Voltage values for h=1
        v_2 : array
            Voltage values for h=2
        phase : array
            Phase values
        time : array or None
            Time values (if given)
        sync_momentum : array or None
            Momentum values (if given, time also must be given)
        t_range : list or None
            Time range injection (if given, time also must be given)
        windowed : bool
            If True, the voltage programs will be 0 until the injection time.
        """
        
        self.x = None
        self.y = None
        self.v1 = v_1
        self.v2 = v_2
        self.dphi = phase
        self.time = time
        self.sync_momentum = sync_momentum
        self.t_range = t_range
        self.windowed = windowed


        assert(len(v_1) == len(v_2) == len(phase), "v_1, v_2 and phase must have the same length")


        # Plot them in separate figures 
        # such that both graphs can have individual events

        self.fig1 = plt.figure(figsize=(15,9))
        self.fig2 = plt.figure(figsize=(15,9))
        self.fig3 = plt.figure(figsize=(15,9))
        
        self.ax = self.fig1.add_subplot(111)
        self.ax2 = self.fig2.add_subplot(111)
        self.ax3 = self.fig3.add_subplot(111)
        
        
        if self.sync_momentum is not None:
            if self.time is None:
                # Create an error and stop the program
                raise ValueError('If sync_momentum is given, time must be given')

            self.fig4 = plt.figure(figsize=(6,9))
            self.ax4 = self.fig4.add_subplot(111)
            self.ax4.plot(self.time, self.sync_momentum, 'r', label = 'Momentum program')
            self.ax4.grid()
            self.ax4.set_xlabel('Time [ms]')
            self.ax4.set_ylabel('Momentum [eV/c]')
            self.ax4.set_title('Momentum program')
            self.ax4.axvline(self.t_range[0], color = 'r', linestyle = '--')
            self.ax4.axvline(self.t_range[1], color = 'r', linestyle = '--')
            self.ax.axvline(self.t_range[0], color = 'r', linestyle = '--')
            self.ax.axvline(self.t_range[1], color = 'r', linestyle = '--')
            self.ax2.axvline(self.t_range[0], color = 'r', linestyle = '--')
            self.ax2.axvline(self.t_range[1],color = 'r', linestyle = '--')
            self.ax3.axvline(self.t_range[0],color = 'r', linestyle = '--')
            self.ax3.axvline(self.t_range[1],color = 'r', linestyle = '--')


            index1 = np.where(self.time <= self.t_range[0])[0][-1]

            if self.windowed:
                # Set all values of voltage arrays to 0 before index1
                self.v1[:index1] = 0
                self.v2[:index1] = 0
                


        if self.time is not None:
            self.xlab = 'Time [ms]'
            self.ax.plot(self.time, self.v1, 'g',label = 'h=1' )
            self.ax2.plot(self.time, self.v2, 'g', label = 'h=2')
            self.ax3.plot(self.time, self.dphi, 'b', label = 'Phase')

            # Set the minor ticks to be every 20
            self.minor_x = Multiple(denominator=5, number=100)
            self.ax.xaxis.set_minor_locator(self.minor_x.locator())
            self.ax2.xaxis.set_minor_locator(self.minor_x.locator())
            self.ax3.xaxis.set_minor_locator(self.minor_x.locator())
            
        else:
            self.xlab = 'Turn number'
            self.ax.plot(v_1, 'g',label = 'h=1' ) 
            self.ax2.plot(v_2,  'g', label = 'h=2')
            self.ax3.plot(phase, 'b', label = 'Phase')

        self.ylab = 'Voltage [V]'
        self.plab = 'Phase [rad]'
        
        # Plot the date with the y axis ranging from 0 to 1.5 times the max

        ## Voltage h=1
        self.ax.set_title('Voltage program for h=1')
        
        self.ax.set_ylim(-0.1*np.max(v_1), 1.5*np.max(v_1))
        self.ax.set_title('h=1')
        self.ax.set_xlabel(self.xlab)
        self.ax.set_ylabel(self.ylab)
    
        #make a grid
        self.ax.grid( which='major', color='k', linestyle='--')
        self.ax.grid( which='minor', color='k', linestyle=':')





        ## Voltage h=2  
        self.ax2.set_title('Voltage program for h=2')
        
        self.ax2.set_ylim(-0.1*np.max(v_2), 1.5*np.max(v_2))
        self.ax2.set_title('h=2')
        self.ax2.set_xlabel(self.xlab)
        self.ax2.set_ylabel(self.ylab)
        self.ax2.grid(  which='major', color='k', linestyle='--')
        self.ax2.grid( which='minor', color='k', linestyle=':')

        
        ## Phase
        self.major_p = Multiple(denominator=denominator_phase, number=np.pi)
        self.minor_p = Multiple(denominator=denominator_phase*4, number=np.pi)
        self.ax3.set_title('Phase difference program')
        
        self.ax3.set_ylim(-1.2*np.pi, 1.2*np.pi)
        self.ax3.set_title('Phase')
        self.ax3.set_xlabel(self.xlab)
        self.ax3.set_ylabel(self.plab)
        self.ax3.yaxis.set_major_locator(self.major_p.locator())
        self.ax3.yaxis.set_minor_locator(self.minor_p.locator())
        # Add x axis minor ticks every 20 ms 
        
        self.ax3.grid(  which='major', color='k', linestyle='--')
        self.ax3.grid(which='minor', color='k', linestyle=':')

        # Keep the initial values for tracking
        self.v1_prev = np.copy(self.v1)
        self.v2_prev = np.copy(self.v2)
        self.dphi_prev = np.copy(self.dphi)
        


        self.fig1.canvas.mpl_connect('button_press_event', self.onpick1)
        self.fig2.canvas.mpl_connect('button_press_event', self.onpick2)
        self.fig3.canvas.mpl_connect('button_press_event', self.onpickphase)




        plt.show(block=True)


    def onpick1(self,event):
        self.x1 , self.y1 =  event.xdata, event.ydata
        
        if self.time is not None:
            print('Selected point (time): ', self.x1, self.y1)
        else:
            print('Selected point (h=1): ', np.round(self.x1), self.y1)
        self.comp_v_1()
        self.update_plot1()

    
    def onpick2(self,event):
        self.x2 , self.y2 =  event.xdata, event.ydata

    # if isinstance(artist, plt.AxesImage):
        print('Selected point (h=2): ', np.round(self.x2), self.y2)
        self.comp_v_2()
        self.update_plot2()

    def onpickphase(self, event):
        self.x3 , self.y3 =  event.xdata, event.ydata

        print('Selected point (phase): ', np.round(self.x3, 2), self.y3)
        self.comp_phase()
        self.update_plotphase()

        

    
    def comp_v_1(self):
        # Find the closest index to the x data 
        if self.time is not None:
            index = np.where(self.time <= self.x1)[0][-1]
        else:
            index = np.round(self.x1).astype(int)

        if self.y1 < 0:
            self.y1 = 0
            self.v1[index:] = self.y1
        else:
            self.v1[index:] = self.y1

    def comp_v_2(self):
        # Find the closest index to the x data
        if self.time is not None:
            index = np.where(self.time <= self.x2)[0][-1]
        else:
            index = np.round(self.x2).astype(int)

        if self.y2 < 0:
            self.y2 = 0
            self.v2[index:] = self.y2
        else: 
            self.v2[index:] = self.y2
        
    
    def comp_phase(self):
        # Find the closest index to the x data
        if self.time is not None:
            if self.x3<=0: 
                index = 0
            else: 
                index = np.where(self.time <= self.x3)[0][-1]
        else:
            index = np.round(self.x3).astype(int)

        self.dphi[index:] = self.y3


    def update_plot1(self):
        self.ax.clear()
        if self.time is not None:
            self.ax.plot(self.time, self.v1_prev, 'k-', label = 'h=1 (prev)')
            self.ax.plot(self.time, self.v1, 'g', label = 'h=1')
            self.ax.axvline(self.t_range[0], color = 'r', linestyle = '--')
            self.ax.axvline(self.t_range[1], color = 'r', linestyle = '--')
        else:
            self.ax.plot(self.v1_prev, 'k-', label = 'h=1 (prev)')
            self.ax.plot(self.v1,'g' ,label = 'h=1')

        self.ax.set_ylim(-0.1*np.max(self.v1), 1.5*np.max(self.v1))
        self.ax.set_xlabel(self.xlab)
        self.ax.set_ylabel(self.ylab)
        self.ax.xaxis.set_minor_locator(self.minor_x.locator())
        self.ax.legend()
        self.ax.grid( which='major', color='k', linestyle='--')
        self.ax.grid( which='minor', color='k', linestyle=':')
        self.fig1.canvas.draw_idle()

    def update_plot2(self):
        self.ax2.clear()
        if self.time is not None:
            self.ax2.plot(self.time, self.v2_prev, 'r-', label = 'h=2 (prev)')
            self.ax2.plot(self.time, self.v2, 'b', label = 'h=2')
            self.ax2.axvline(self.t_range[0], color = 'r', linestyle = '--')
            self.ax2.axvline(self.t_range[1], color = 'r', linestyle = '--')
        else:
            self.ax2.plot(self.v2_prev, 'r-', label = 'h=2 (prev)')
            self.ax2.plot(self.v2,'b' ,label = 'h=2')

        self.ax2.set_ylim(-0.1*np.max(self.v2), 1.5*np.max(self.v2))
        self.ax2.set_xlabel(self.xlab)
        self.ax2.set_ylabel(self.ylab)
        self.ax2.xaxis.set_minor_locator(self.minor_x.locator())
        self.ax2.legend()
        self.ax2.grid( which='major', color='k', linestyle='--')
        self.ax2.grid( which='minor', color='k', linestyle=':')
        self.fig2.canvas.draw_idle()

    def update_plotphase(self):
        self.ax3.clear()

        if self.time is not None:
            self.ax3.plot(self.time, self.dphi_prev, 'm-', label = 'Phase (prev)')
            self.ax3.plot(self.time, self.dphi, 'c', label = 'Phase')
            self.ax3.axvline(self.t_range[0], color = 'r', linestyle = '--')
            self.ax3.axvline(self.t_range[1], color = 'r', linestyle = '--')
        else:
            self.ax3.plot(self.dphi_prev, 'm-', label = 'Phase (prev)')
            self.ax3.plot(self.dphi,'c' ,label = 'Phase')

        self.ax3.set_ylim(-1.2*np.pi, 1.2*np.pi)
        self.ax3.set_xlabel(self.xlab)
        self.ax3.set_ylabel(self.plab)
        self.ax3.yaxis.set_major_locator(self.major_p.locator())
        self.ax3.yaxis.set_minor_locator(self.minor_p.locator())
        self.ax3.xaxis.set_minor_locator(self.minor_x.locator())
        self.ax3.legend()
        self.ax3.grid( which='major', color='k', linestyle='--')
        self.ax3.grid( which='minor', color='k', linestyle=':')
    
        self.fig3.canvas.draw_idle()

    

    
    def save_data(self,name):
        # Check if there is a v_programe folder 
        if self.time is not None:
            if not os.path.exists('v_programs_dt'):
                os.makedirs('v_programs_dt')


            np.save('v_programs_dt/'+name+'.npy', [self.v1, self.v2, self.dphi,self.time])
        
        else: 
            if not os.path.exists('v_programs'):
                os.makedirs('v_programs')
        

            np.save('v_programs/'+name+'.npy', [self.v1, self.v2, self.dphi])


class Waterfaller:
    def __init__(self, Profile, curr_turn, sampling_d_turns =100, n_turns = None, traces = 10000, USE_GPU = False):
        """"
        Makes a waterfall plot by taking the traces of the profile
        and plotting them in a waterfall plot

        Parameters
        ----------
        Profile : Profile
            Profile object
        
        curr_turn : int
            Current turn
        
        sampling_d_turns : int
            Sampling rate of the waterfall plot
        
        n_turns : int
            Number of total turns after which plotting is done,
            it should only be given when traces is None
            
        traces : int
            Number of traces to plot
        
        """
        self.Profile = Profile
        self.USE_GPU = USE_GPU
        if self.USE_GPU:
            self.data = [(self.Profile.bin_centers.get(),self.Profile.n_macroparticles.get())]
        else:
            self.data = [(self.Profile.bin_centers,self.Profile.n_macroparticles)]
        # self.extent = [np.min(self.Profile.bin_centers.get()) , np.max(self.Profile.bin_centers.get())]
        self.traces = traces
        self.sampling = sampling_d_turns
        self.n_turns = n_turns
        self.counter = 1
        self.globalcounter = curr_turn
        self.z_s = [self.globalcounter]
        

        # Fail safe situation due to operator error 
        if n_turns is None and traces is None:
            self.traces = 10000
        

    def update(self,Profile,curr_turn):
        self.globalcounter = curr_turn
        

        if self.counter % self.sampling == 0:
            if self.USE_GPU:
                self.data.append((Profile.bin_centers.get(),Profile.n_macroparticles.get()))
            else:
                self.data.append((Profile.bin_centers,Profile.n_macroparticles))
            self.z_s.append(self.globalcounter)

            
        

        if self.globalcounter == self.n_turns:
            if self.USE_GPU:
                self.data.append((Profile.bin_centers.get(),Profile.n_macroparticles.get()))
            else:
                self.data.append((Profile.bin_centers,Profile.n_macroparticles))
            self.z_s.append(self.globalcounter)
            self.plot()

        

        
        if self.counter == self.traces:
            self.plot()
            if self.USE_GPU:
                self.data = [(self.Profile.bin_centers.get(),self.Profile.n_macroparticles.get())]
            else:
                self.data = [(self.Profile.bin_centers,self.Profile.n_macroparticles)]
            self.counter = 1
            self.z_s = [self.globalcounter]

        self.counter +=1
        


    
    def plot(self):
        # Rearange the data such that the bin_centers are the x_axis
        # the different timesteps are in the y_axis
        # and the n_macroparticles are in the color scale
        print('Plotting waterfall plot')

        z = np.array(self.z_s)
        
        x = np.array(self.data[0][0])
        
        # xx,zz = np.meshgrid(x,z)

        #Stack the n_macroparticles on the z axis
        for i in range(len(self.data)):
            if i == 0:
                y = np.array(self.data[i][1])
            else:
                y = np.vstack((y,np.array(self.data[i][1])))

        fig = plt.figure()
    

        ax = fig.add_subplot(111)
       

        p = ax.pcolormesh(x, z, y, cmap='viridis',linewidth = 0, antialiased=False)

        

        ax.set_label('n_macroparticles')
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Turn number')
        ax.set_title('Waterfall plot')
        # add a colorbar with the label "n_macroparticles"
        fig.colorbar(p, ax=ax)


        plt.savefig('./gpu_output_files/EX_02_fig/waterfall_plot_'+str(self.globalcounter)+'.png')


class ProgramDefOpt(object):
    def __init__(self,v_1,v_2,time=None, sync_momentum = None, t_range = None, windowed = False, denominator_phase = 4):
        """
        This is a class to plot the voltage (not phase as it is used to optimize) 
        over time and iteratively updates them depending on the user clicks on the 
        figure.

        Intructions: 
        Whatever point is clicked, it will update the value of that point
        at that given time and all subsequent points.
        The program will be closed when the user closes a figure.

        To set the voltage to 0, it is sufficient to click a value below the 
        x axis. This will set all subsequent values to 0.


        Parameters
        ----------
        v_1 : array
            Voltage values for h=1
        v_2 : array
            Voltage values for h=2
        phase : array
            Phase values
        time : array or None
            Time values (if given)
        sync_momentum : array or None
            Momentum values (if given, time also must be given)
        t_range : list or None
            Time range injection (if given, time also must be given)
        windowed : bool
            If True, the voltage programs will be 0 until the injection time.
        """
        
        self.x = None
        self.y = None
        self.v1 = v_1
        self.v2 = v_2
        self.time = time
        self.sync_momentum = sync_momentum
        self.t_range = t_range
        self.windowed = windowed


        assert len(v_1) == len(v_2) , "v_1, v_2 and phase must have the same length"


        # Plot them in separate figures 
        # such that both graphs can have individual events

        self.fig1 = plt.figure(figsize=(15,9))
        self.fig2 = plt.figure(figsize=(15,9))
        
        self.ax = self.fig1.add_subplot(111)
        self.ax2 = self.fig2.add_subplot(111)
        
        
        if self.sync_momentum is not None:
            if self.time is None:
                # Create an error and stop the program
                raise ValueError('If sync_momentum is given, time must be given')

            self.fig4 = plt.figure(figsize=(6,9))
            self.ax4 = self.fig4.add_subplot(111)
            self.ax4.plot(self.time, self.sync_momentum, 'r', label = 'Momentum program')
            self.ax4.grid()
            self.ax4.set_xlabel('Time [s]')
            self.ax4.set_ylabel('Momentum [eV/c]')
            self.ax4.set_title('Momentum program')
            if self.t_range is not None:

                self.ax4.axvline(self.t_range[0], color = 'r', linestyle = '--')
                self.ax4.axvline(self.t_range[1], color = 'r', linestyle = '--')
                self.ax.axvline(self.t_range[0], color = 'r', linestyle = '--')
                self.ax.axvline(self.t_range[1], color = 'r', linestyle = '--')
                self.ax2.axvline(self.t_range[0], color = 'r', linestyle = '--')
                self.ax2.axvline(self.t_range[1],color = 'r', linestyle = '--')


                index1 = np.where(self.time <= self.t_range[0])[0][-1]

                if self.windowed:
                    # Set all values of voltage arrays to 0 before index1
                    self.v1[:index1] = 0
                    self.v2[:index1] = 0
                


        if self.time is not None:
            self.xlab = 'Time [s]'
            self.ax.plot(self.time, self.v1, 'g',label = 'h=1' )
            self.ax2.plot(self.time, self.v2, 'g', label = 'h=2')

            # Set the minor ticks to be every 20
            self.minor_x = Multiple(denominator=5, number=100)
            self.ax.xaxis.set_minor_locator(self.minor_x.locator())
            self.ax2.xaxis.set_minor_locator(self.minor_x.locator())
            
        else:
            self.xlab = 'Turn number'
            self.ax.plot(v_1, 'g',label = 'h=1' ) 
            self.ax2.plot(v_2,  'g', label = 'h=2')

        self.ylab = 'Voltage [V]'
        
        # Plot the date with the y axis ranging from 0 to 1.5 times the max

        ## Voltage h=1
        self.ax.set_title('Voltage program for h=1')
        
        self.ax.set_ylim(-0.1*np.max(v_1), 1.5*np.max(v_1))
        self.ax.set_title('h=1')
        self.ax.set_xlabel(self.xlab)
        self.ax.set_ylabel(self.ylab)
    
        #make a grid
        self.ax.grid( which='major', color='k', linestyle='--')
        self.ax.grid( which='minor', color='k', linestyle=':')





        ## Voltage h=2  
        self.ax2.set_title('Voltage program for h=2')
        
        self.ax2.set_ylim(-0.1*np.max(v_2), 1.5*np.max(v_2))
        self.ax2.set_title('h=2')
        self.ax2.set_xlabel(self.xlab)
        self.ax2.set_ylabel(self.ylab)
        self.ax2.grid(  which='major', color='k', linestyle='--')
        self.ax2.grid( which='minor', color='k', linestyle=':')

        
        ## Phase
        # Add x axis minor ticks every 20 ms 
        

        # Keep the initial values for tracking
        self.v1_prev = np.copy(self.v1)
        self.v2_prev = np.copy(self.v2)
        


        self.fig1.canvas.mpl_connect('button_press_event', self.onpick1)
        self.fig2.canvas.mpl_connect('button_press_event', self.onpick2)


        self.fig1.canvas.mpl_connect('close_event', self.close_plots)
        self.fig2.canvas.mpl_connect('close_event', self.close_plots)



        plt.show()


    def onpick1(self,event):
        self.x1 , self.y1 =  event.xdata, event.ydata
        
        if self.time is not None:
            print('Selected point (time): ', self.x1, self.y1)
        else:
            print('Selected point (h=1): ', np.round(self.x1), self.y1)
        self.comp_v_1()
        self.update_plot1()

    
    def onpick2(self,event):
        self.x2 , self.y2 =  event.xdata, event.ydata

    # if isinstance(artist, plt.AxesImage):
        print('Selected point (h=2): ', np.round(self.x2), self.y2)
        self.comp_v_2()
        self.update_plot2()

    
    def comp_v_1(self):
        # Find the closest index to the x data 
        if self.time is not None:
            index = np.where(self.time <= self.x1)[0][-1]
        else:
            index = np.round(self.x1).astype(int)

        if self.y1 < 0:
            self.y1 = 0
            self.v1[index:] = self.y1
        else:
            self.v1[index:] = self.y1

    def comp_v_2(self):
        # Find the closest index to the x data
        if self.time is not None:
            index = np.where(self.time <= self.x2)[0][-1]
        else:
            index = np.round(self.x2).astype(int)

        if self.y2 < 0:
            self.y2 = 0
            self.v2[index:] = self.y2
        else: 
            self.v2[index:] = self.y2
        


    def update_plot1(self):
        self.ax.clear()
        if self.time is not None:
            self.ax.plot(self.time, self.v1_prev, 'k-', label = 'h=1 (prev)')
            self.ax.plot(self.time, self.v1, 'g', label = 'h=1')
            if self.t_range is not None:
                self.ax.axvline(self.t_range[0], color = 'r', linestyle = '--')
                self.ax.axvline(self.t_range[1], color = 'r', linestyle = '--')
        else:
            self.ax.plot(self.v1_prev, 'k-', label = 'h=1 (prev)')
            self.ax.plot(self.v1,'g' ,label = 'h=1')

        self.ax.set_ylim(-0.1*np.max(self.v1), 1.5*np.max(self.v1))
        self.ax.set_xlabel(self.xlab)
        self.ax.set_ylabel(self.ylab)
        self.ax.xaxis.set_minor_locator(self.minor_x.locator())
        self.ax.legend()
        self.ax.grid( which='major', color='k', linestyle='--')
        self.ax.grid( which='minor', color='k', linestyle=':')
        self.fig1.canvas.draw_idle()

    def update_plot2(self):
        self.ax2.clear()
        if self.time is not None:
            self.ax2.plot(self.time, self.v2_prev, 'r-', label = 'h=2 (prev)')
            self.ax2.plot(self.time, self.v2, 'b', label = 'h=2')
            if self.t_range is not None:
                self.ax2.axvline(self.t_range[0], color = 'r', linestyle = '--')
                self.ax2.axvline(self.t_range[1], color = 'r', linestyle = '--')
        else:
            self.ax2.plot(self.v2_prev, 'r-', label = 'h=2 (prev)')
            self.ax2.plot(self.v2,'b' ,label = 'h=2')

        self.ax2.set_ylim(-0.1*np.max(self.v2), 1.5*np.max(self.v2))
        self.ax2.set_xlabel(self.xlab)
        self.ax2.set_ylabel(self.ylab)
        self.ax2.xaxis.set_minor_locator(self.minor_x.locator())
        self.ax2.legend()
        self.ax2.grid( which='major', color='k', linestyle='--')
        self.ax2.grid( which='minor', color='k', linestyle=':')
        self.fig2.canvas.draw_idle()



    def close_plots(self,event):
        # When the figure is closed, return the timeseries
        # Force them all to close
        plt.close('all')

        # Close remaining windows
        plt.close('all')

        

    
    def save_data(self,name):
        # Check if there is a v_programe folder 
        if self.time is not None:
            if not os.path.exists('v_programs_dt'):
                os.makedirs('v_programs_dt')


            np.save('v_programs_dt/'+name+'.npy', [v1, v2,self.time])
        
        else: 
            if not os.path.exists('v_programs'):
                os.makedirs('v_programs')
        

            np.save('v_programs/'+name+'.npy', [v1, v2])
    
    def save_data_full(self,name, phase):
        # Check if there is a v_programe folder
        if self.time is not None:
            if not os.path.exists('v_programs_dt'):
                os.makedirs('v_programs_dt')


            np.save('v_programs_dt/'+name+'.npy', [v1, v2, phase,self.time])
        
        else: 
            if not os.path.exists('v_programs'):
                os.makedirs('v_programs')
        

            np.save('v_programs/'+name+'.npy', [v1, v2, phase])
        return
    
class tolSetter(object):
    """
    This class allows you to set the tolerance of the peak finder
    from a GUI showing the gradient of the stable phase
    """
    def __init__(self, phase_diff):

        self.phase_diff = phase_diff   
        self.fig1 = plt.figure(figsize=(15,9))
        
        self.ax = self.fig1.add_subplot(111)
        
        
    

        
        self.xlab = 'Turn number'
        self.ax.plot(self.phase_diff, 'g',label = 'h=1' ) 

        self.ylab = 'Synchronous Phase [rad]'
        
        # Plot the date with the y axis ranging from 0 to 1.5 times the max

        ## Voltage h=1
        self.ax.set_title('Synchronous Phase for h=1')
        
        self.ax.set_xlabel(self.xlab)
        self.ax.set_ylabel(self.ylab)
    
        #make a grid
        self.ax.grid( which='major', color='k', linestyle='--')
        self.ax.grid( which='minor', color='k', linestyle=':')


        self.fig1.canvas.mpl_connect('button_press_event', self.onpick1)


        self.fig1.canvas.mpl_connect('close_event', self.close_plots)


        plt.show()

    def onpick1(self,event):
        self.x , self.y=  event.xdata, event.ydata
        
        if self.time is not None:
            print('Selected point (time): ', self.x1, self.y1)
        else:
            print('Selected point (h=1): ', np.round(self.x1), self.y1)
        self.update_plot()

    def update_plot(self):

        self.ax.axhline(self.y, '-r')
        self.ax.plot(self.phase_diff, 'g')
        self.ax.set_xlabel(self.xlab)
        self.ax.set_ylabel(self.ylab)
        self.ax.set_title('Synchronous Phase for h=1')
        self.ax.grid( which='major', color='k', linestyle='--')
        self.ax.grid( which='minor', color='k', linestyle=':')

        self.fig1.canvas.draw_idle()


    def close_plots(self,event):
        # When the figure is closed, return the timeseries
        # Force them all to close
        plt.close('all')


    
class WaterfallerOpt:
    def __init__(self, Profile, curr_turn, sampling_d_turns =100, n_turns = None, traces = 10000, USE_GPU = False):
        """"
        Makes a waterfall plot by taking the traces of the profile
        and plotting them in a waterfall plot

        Parameters
        ----------
        Profile : Profile
            Profile object
        
        curr_turn : int
            Current turn
        
        sampling_d_turns : int
            Sampling rate of the waterfall plot
        
        n_turns : int
            Number of total turns after which plotting is done,
            it should only be given when traces is None
            
        traces : int
            Number of traces to plot
        
        """
        self.Profile = Profile
        self.USE_GPU = USE_GPU
        if self.USE_GPU:
            self.data = [(self.Profile.bin_centers.get(),self.Profile.n_macroparticles.get())]
        else:
            self.data = [(self.Profile.bin_centers,self.Profile.n_macroparticles)]
        # self.extent = [np.min(self.Profile.bin_centers.get()) , np.max(self.Profile.bin_centers.get())]
        self.traces = traces
        self.sampling = sampling_d_turns
        self.n_turns = n_turns
        self.counter = 1
        self.globalcounter = curr_turn
        self.z_s = [self.globalcounter]
        

        # Fail safe situation due to operator error 
        if n_turns is None and traces is None:
            self.traces = 10000
        

    def update(self,Profile,curr_turn):
        self.globalcounter = curr_turn
        

        if self.globalcounter % self.sampling == 0:
            if self.USE_GPU:
                self.data.append((Profile.bin_centers.get(),Profile.n_macroparticles.get()))
            else:
                self.data.append((Profile.bin_centers,Profile.n_macroparticles))
            self.z_s.append(self.globalcounter)

            
        

        if self.globalcounter == self.n_turns:
            if self.USE_GPU:
                self.data.append((Profile.bin_centers.get(),Profile.n_macroparticles.get()))
            else:
                self.data.append((Profile.bin_centers,Profile.n_macroparticles))
            self.z_s.append(self.globalcounter)
            self.plot()
            return self.data

        

        
        if self.globalcounter == self.traces:
            self.plot()
            if self.USE_GPU:
                self.data = [(self.Profile.bin_centers.get(),self.Profile.n_macroparticles.get())]
            else:
                self.data = [(self.Profile.bin_centers,self.Profile.n_macroparticles)]
            self.counter = 1
            self.z_s = [self.globalcounter]

        self.counter +=1
        
        


    
    def plot(self):
        # Rearange the data such that the bin_centers are the x_axis
        # the different timesteps are in the y_axis
        # and the n_macroparticles are in the color scale
        # print('Plotting waterfall plot')

        z = np.array(self.z_s)
        x = np.array(self.data[0][0])
        # xx,zz = np.meshgrid(x,z)

        #Stack the n_macroparticles on the z axis
        for i in range(len(self.data)):
            if i == 0:
                y = np.array(self.data[i][1])
            else:
                y = np.vstack((y,np.array(self.data[i][1])))

        fig = plt.figure()
    

        ax = fig.add_subplot(111)
       

        p = ax.pcolormesh(x, z, y, cmap='viridis',linewidth = 0, antialiased=False)

        

        ax.set_label('n_macroparticles')
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Turn number')
        ax.set_title('Waterfall plot')
        # add a colorbar with the label "n_macroparticles"
        fig.colorbar(p, ax=ax)


        plt.savefig('./gpu_output_files/EX_02_fig/waterfall_plot_'+str(self.globalcounter)+'.png')


        plt.close('all')

if __name__ == '__main__':
    v1 = np.ones(30)*200
    v2 = np.ones(30)*30
    phase = np.ones(30)*0.5*np.pi

    plots = ProgramDef(v1,v2, phase)

    print(v1[20], v2[20], phase[20])


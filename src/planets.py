"""
Harvard IACS Masters Thesis
Trajectories for Known Asteroids

Michael S. Emanuel
Fri Aug 23 16:13:28 2019
"""

# Library imports
import rebound
import matplotlib.pyplot as plt
from datetime import datetime
import shutil
import argparse
from typing import List

# Local imports
from astro_utils import  mjd_to_datetime
from rebound_utils import make_sim, extend_sim, make_archive, test_integration
from utils import plot_style

# ********************************************************************************************************************* 
# Set plot style
plot_style()

# ********************************************************************************************************************* 
# Collections of objects

# The sun and 8 planets
planets = [
    'Sun', 'Mercury Barycenter', 'Venus Barycenter', 
    'Earth', 'Moon',
    'Mars Barycenter',  'Jupiter Barycenter', 'Saturn Barycenter', 
    'Uranus Barycenter', 'Neptune Barycenter']
object_names_planets = planets
# Test the barycenters of other planets; skip the moon because this model can't handle it!
test_objects_planets = [nm for nm in object_names_planets if nm != 'Moon']

# The sun, 8 planets, and the most significant moons
# See https://en.wikipedia.org/wiki/List_of_Solar_System_objects_by_size
moons = [
    'Sun', 
    'Mercury', 
    'Venus', 
    'Earth', 'Moon', 
    'Mars Barycenter',
    'Jupiter', 'Io', 'Europa', 'Ganymede', 'Callisto',
    'Saturn', 'Mimas', 'Enceladus', 'Tethys', 'Dione', 'Rhea', 'Titan', 'Iapetus', 'Phoebe',
    'Uranus', 'Ariel', 'Umbriel', 'Titania', 'Oberon', 'Miranda',
    'Neptune', 'Triton', 'Proteus',
    'Pluto', 'Charon']
object_names_moons = moons
# These are the objects tested for the moon integration
test_objects_moons = [
    'Sun', 'Mercury', 'Venus', 'Earth', 'Mars Barycenter', 
    'Jupiter', 'Saturn', 'Uranus', 'Neptune']

# Planet barycenters and selected dwarf planets above 1E-10 solar masses
# block 1: mass above 1E-9 solar masses; 4 of the 5 IAU recognized
dwarfs_09 = ['Pluto Barycenter', 'Eris', 'Makemake', 'Haumea']
# block 2: mass above 1E-10 solar masses
dwarfs_10 = [
    '2007 OR10', 'Quaoar', 'Hygiea', 'Ceres', 'Orcus', 
    'Salacia', 'Varuna', 'Varda', 'Vesta', 'Pallas']
# block 3: mass above 1E-11 solar masses
dwarfs_11 = [
    '229762', '2002 UX25', '2002 WC19', 'Davida', 'Interamnia', 
    'Eunomia', '2004 UX10', 'Juno', 'Psyche', '52 Europa']

# Selected dwarfs
dwarfs = dwarfs_09 + dwarfs_10
# Object collection for dwarfs integration: planet barycenters + selected dwarfs
object_names_dwarfs = object_names_planets + dwarfs
# Test objects for dwarfs integration - same as for planets
test_objects_dwarfs = test_objects_planets

# Objects in collection 'all'
object_names_all = object_names_moons + dwarfs
test_objects_all = test_objects_moons

# Shared collection of test asteroids to integrate
test_asteroids = [
    'Ceres', 'Pallas', 'Juno', 'Vesta', 'Iris',
    'Hygiea', 'Egeria', 'Eunomia', 'Psyche', 'Fortuna']
test_objects_asteroids = ['Earth'] + test_asteroids

# ********************************************************************************************************************* 
def make_sim_planets(epoch: datetime, integrator='ias15', steps_per_day: int = 256):
    """Create a simulation with the sun and 8 planets at the specified time"""
    # Arguments for make_sim
    sim_name = 'planets'    
    object_names = object_names_planets
    save_file = False

    # Build a simulation with the selected objects
    sim = make_sim(sim_name=sim_name, object_names=object_names, epoch=epoch, 
                   integrator=integrator, steps_per_day=steps_per_day, save_file=save_file)

    return sim

# ********************************************************************************************************************* 
def make_sim_moons(epoch: datetime, integrator='ias15', steps_per_day: int = 16):
    """Create a simulation with the sun and 8 planets plus selected moons"""
    # Arguments for make_sim
    sim_name = 'moons'
    object_names = object_names_moons
    save_file = False

    # Build a simulation with the selected objects
    sim = make_sim(sim_name=sim_name, object_names=object_names, epoch=epoch, 
                   integrator=integrator, steps_per_day=steps_per_day, save_file=save_file)

    return sim

# ********************************************************************************************************************* 
def make_sim_dwarfs(epoch: datetime, integrator='ias15', steps_per_day: int = 16):
    """Create a simulation with the sun, 8 planets and selected dwarf planets"""
    # Arguments for make_sim
    sim_name = 'dwarfs'
    object_names = object_names_dwarfs
    save_file = False

    # Build a simulation with the selected objects
    sim = make_sim(sim_name=sim_name, object_names=object_names, epoch=epoch, 
                   integrator=integrator, steps_per_day=steps_per_day, save_file=save_file)

    return sim

# ********************************************************************************************************************* 
def make_sim_all(epoch: datetime, integrator='ias15', steps_per_day: int = 16):
    """Create a simulation with all the massive objects (planets, moons, dwarf planets)"""
    # Arguments for make_sim
    sim_name = 'all'
    object_names = object_names_all
    save_file = False

    # Build a simulation with the selected objects
    sim = make_sim(sim_name=sim_name, object_names=object_names, epoch=epoch, 
                   integrator=integrator, steps_per_day=steps_per_day, save_file=save_file)

    return sim

# ********************************************************************************************************************* 
def run_one_sim(sim_name: str, sim_epoch: rebound.Simulation, object_names: List[str],
                epoch: datetime, dt0: datetime, dt1: datetime, 
                time_step: int, save_step: int, steps_per_day: int):
    """Run one simulation, saving it to a simulation archive"""
    integrator = sim_epoch.integrator
    fname = f'../data/planets/sim_{sim_name}_2000-2040_{integrator}_sf{steps_per_day}.bin'
    fname_gen = f'../data/planets/sim_{sim_name}_2000-2040.bin'
    sa = make_archive(fname_archive=fname, sim_epoch=sim_epoch, object_names=object_names,
                      epoch=epoch, dt0=dt0, dt1=dt1, 
                      time_step=time_step, save_step=save_step,
                      progbar=True)
    # Copy file to generically named one
    shutil.copy(fname, fname_gen)
    
    return sa

# ********************************************************************************************************************* 
def add_test_asteroid_names(object_names: List[str], test_asteroids: List[str]) -> List[str]:
    """Augment a list of object names to include the test asteroids"""
    object_names_new = object_names.copy()
    for nm in test_asteroids:
        if nm not in object_names:
            object_names_new.append(nm)
    return object_names_new

# ********************************************************************************************************************* 
def set_test_asteroid_masses(sim: rebound.Simulation, 
                             object_names_orig: List[str], 
                             test_asteroids: List[str]) -> None:
    """Zero out the masses of test asteroids to zero if they were not originally added as massive objects"""
    # List of objects to zero
    names_zero = [nm for nm in test_asteroids if nm not in object_names_orig]
    # Set masses of these objects to zero
    for nm in names_zero:
        sim.particles[nm].m = 0.0

# ********************************************************************************************************************* 
def add_test_asteroids(sim: rebound.Simulation, 
                       object_names: List[str],
                       test_asteroids: List[str],
                       epoch: datetime) -> None:
    """Augment a rebound Simulation to include test asteroids; modifies the simulation in place"""
    # Get the names of the test asteroids to add
    object_names_new = add_test_asteroid_names(object_names=object_names, test_asteroids=test_asteroids)

    # Add these new objects using extend_sim
    sim = extend_sim(sim, object_names_new=object_names_new, epoch=epoch)

    # Zero out the test asteroid masses
    set_test_asteroid_masses(sim=sim, object_names_orig=object_names_planets, test_asteroids=test_asteroids)    

# ********************************************************************************************************************* 
def test_one_sim(sa: rebound.SimulationArchive, test_objects: List[str],
                 sim_name: str, test_name: str, sim_long_name: str,
                 integrator: str, steps_per_day: int,
                 make_plot_internal: bool=True, verbose_internal: bool=False,
                 make_plot_asteroids: bool=True, verbose_asteroids: bool=False):
    """Test one integration on a set of test objects and the common asteroid test objects."""
    # Status
    print(f'\n***** Testing integration of {sim_long_name}. *****')
    print(f'Integrator = {integrator}, steps per day = {steps_per_day}.')

    # Test against the named objects (internal test)
    test_integration(sa=sa, test_objects=test_objects, 
                     sim_name=sim_name, test_name=test_name, 
                     make_plot=make_plot_internal, verbose=verbose_internal)

    # Test against the asteroid test set
    pos_err, ang_err = \
        test_integration(sa=sa, test_objects=test_objects_asteroids, 
                         sim_name=sim_name, test_name='asteroids', 
                         make_plot=make_plot_asteroids, verbose=verbose_asteroids)
    
    return pos_err, ang_err

# ********************************************************************************************************************* 
def main():
    """Integrate the orbits of the planets and major moons"""
    
    # Process command line arguments
    parser = argparse.ArgumentParser(description='Integrate the orbits of planets and moons.')
    parser.add_argument('type', nargs='?', metavar='type', type=str, default='A',
                        help='type of integration: p- planets; d- dwarfs; m-moons; a-all, '
                             'A-run all 4 strategies')
    parser.add_argument('steps_per_day', nargs='?', metavar='SPD', type=int, default=-1,
                        help='the (max) number of steps per day taken by the integrator')
    parser.add_argument('--test', default=False, action='store_true',
                        help='run in test mode')
    args = parser.parse_args()
    
    # Unpack command line arguments
    integration_type = args.type
    steps_per_day: int = args.steps_per_day
    
    # Flags for planets and moons
    run_planets: bool = integration_type in ('p', 'A')
    run_moons: bool = integration_type in ('m', 'A')
    run_dwarfs: bool = integration_type in ('d', 'A')
    run_all: bool = integration_type in ('a', 'A')

    # Reference epoch for asteroids file
    epoch_mjd: float = 58600.0
    # Convert to a datetime
    epoch: datetime = mjd_to_datetime(epoch_mjd)
    # epoch_dt: datetime = datetime(2019,4,27)
    
    # Start and end times of simulation
    dt0: datetime = datetime(2000, 1, 1)
    dt1: datetime = datetime(2040,12,31)
    
    # Integrator choices
    integrator_planets: str = 'ias15'
    integrator_moons: str = 'ias15'
    integrator_dwarfs: str = 'ias15'
    integrator_all: str = 'ias15'
    
    # Integrator time step
    steps_per_day_planets: int = steps_per_day if steps_per_day >= 0 else 16
    steps_per_day_moons: int = steps_per_day if steps_per_day >= 0 else 16
    steps_per_day_dwarfs: int = steps_per_day if steps_per_day >= 0 else 16
    steps_per_day_all: int = steps_per_day if steps_per_day >= 0 else 16

    # Initial configuration of planets, moons, and dwarfs
    sim_planets = make_sim_planets(epoch=epoch, integrator=integrator_planets, 
                                   steps_per_day=steps_per_day_planets)
    sim_moons = make_sim_moons(epoch=epoch, integrator=integrator_moons, 
                               steps_per_day=steps_per_day_moons)
    sim_dwarfs = make_sim_dwarfs(epoch=epoch, integrator=integrator_dwarfs, 
                                 steps_per_day=steps_per_day_dwarfs)
    sim_all = make_sim_all(epoch=epoch, integrator=integrator_all, 
                           steps_per_day=steps_per_day_all)
    
    # If we are in test mode, add the asteroids to the base simulations
    if args.test:
        add_test_asteroids(sim=sim_planets, 
                           object_names=object_names_planets,
                           test_asteroids=test_asteroids, epoch=epoch)

        add_test_asteroids(sim=sim_moons, 
                           object_names=object_names_moons,
                           test_asteroids=test_asteroids, epoch=epoch)

        add_test_asteroids(sim=sim_dwarfs, 
                           object_names=object_names_dwarfs,
                           test_asteroids=test_asteroids, epoch=epoch)

        add_test_asteroids(sim=sim_all, 
                           object_names=object_names_all,
                           test_asteroids=test_asteroids, epoch=epoch)

    # Shared time_step and save_step
    time_step: int = 1
    save_step: int = 1

    # Integrate the planets from dt0 to dt1
    if run_planets:
        sim_name = 'planets' if not args.test else 'planets_test'
        test_name = 'planets_com'
        sim_long_name = 'sun and 8 planets'

        # Run the planets simulation
        sa_planets = \
            run_one_sim(sim_name=sim_name, sim_epoch=sim_planets, object_names=object_names_planets,
                        epoch=epoch, dt0=dt0, dt1=dt1,
                        time_step=time_step, save_step=save_step, steps_per_day=steps_per_day_planets)

    # Test the planets simulation
    if run_planets and args.test:
        pos_err_planets, ang_err_planets = \
            test_one_sim(sa=sa_planets, test_objects=test_objects_planets,
                         sim_name=sim_name, test_name=test_name, sim_long_name=sim_long_name, 
                         integrator=integrator_planets, steps_per_day=steps_per_day_planets,
                         make_plot_internal=False, verbose_internal=False, 
                         make_plot_asteroids=False, verbose_asteroids=False)
    
    # Integrate the planets and moons from dt0 to dt1
    if run_moons:
        sim_name = 'moons' if not args.test else 'moons_test'
        test_name = 'planets'
        sim_long_name = f'{len(object_names_moons)} objects in solar system: sun, planets, moons'

        # Run the moons simulation
        sa_moons = \
            run_one_sim(sim_name=sim_name, sim_epoch=sim_moons, object_names=object_names_moons,
                        epoch=epoch, dt0=dt0, dt1=dt1,
                        time_step=time_step, save_step=save_step, steps_per_day=steps_per_day_moons)

    # Test the moons simulation        
    if run_moons and args.test:
        pos_err_moons, ang_err_moons = \
            test_one_sim(sa=sa_moons, test_objects=test_objects_moons,
                         sim_name=sim_name, test_name=test_name, sim_long_name=sim_long_name, 
                         integrator=integrator_moons, steps_per_day=steps_per_day_moons,
                         make_plot_internal=True, verbose_internal=False, 
                         make_plot_asteroids=True, verbose_asteroids=False)

       
    # Integrate the planets and dwarf planets from dt0 to dt1
    if run_dwarfs:
        sim_name = 'dwarfs' if not args.test else 'dwarfs_test'
        test_name = 'planets'
        sim_long_name = f'{len(object_names_moons)} objects in solar system: sun, planets, dwarf planets'

        # Run the dwarfs simulation
        sa_dwarfs = \
            run_one_sim(sim_name=sim_name, sim_epoch=sim_dwarfs, object_names=object_names_dwarfs,
                        epoch=epoch, dt0=dt0, dt1=dt1,
                        time_step=time_step, save_step=save_step, steps_per_day=steps_per_day_dwarfs)

    # Test the dwarfs simulation
    if run_dwarfs and args.test:
        pos_err_dwarfs, ang_err_dwarfs = \
            test_one_sim(sa=sa_dwarfs, test_objects=test_objects_dwarfs,
                         sim_name=sim_name, test_name=test_name, sim_long_name=sim_long_name, 
                         integrator=integrator_dwarfs, steps_per_day=steps_per_day_dwarfs,
                         make_plot_internal=True, verbose_internal=False, 
                         make_plot_asteroids=True, verbose_asteroids=False)
       
    # Integrate all the bodies from dt0 to dt1
    if run_all:
        sim_name = 'all' if not args.test else 'all_test'
        test_name = 'planets'
        sim_long_name = f'all {len(object_names_all)} heavy objects in solar system: ' \
                        f'sun, planets, moons, dwarf planets'
        # Run the all simulation
        sa_all = \
            run_one_sim(sim_name=sim_name, sim_epoch=sim_all, object_names=object_names_all,
                        epoch=epoch, dt0=dt0, dt1=dt1,
                        time_step=time_step, save_step=save_step, steps_per_day=steps_per_day_all)

    # Test the all simulation        
    if run_all and args.test:
        pos_err_all, ang_err_all = \
            test_one_sim(sa=sa_all, test_objects=test_objects_all,
                         sim_name=sim_name, test_name=test_name, sim_long_name=sim_long_name, 
                         integrator=integrator_all, steps_per_day=steps_per_day_all,
                         make_plot_internal=True, verbose_internal=False, 
                         make_plot_asteroids=True, verbose_asteroids=False)

    if args.test:
        # Plot position error
        fig, ax = plt.subplots(figsize=[16,10])
        ax.set_title(f'Position Error on 10 Test Asteroids')
        ax.set_ylabel('RMS Position Error in AU')
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0,))
        test_years = list(range(2000,2041))
        if run_planets:
            ax.plot(test_years, pos_err_planets, label='planets', marker='+', color='blue')
        if run_moons:
            ax.plot(test_years, pos_err_moons, label='moons', marker='x', color='green')
        if run_dwarfs:
            ax.plot(test_years, pos_err_dwarfs, label='dwarfs', marker='o', color='red')
        ax.grid()
        ax.legend()
        fig.savefig(fname=f'../figs/integration_test/planets/sim_ang_error_comp.png', bbox_inches='tight')
    
        # Plot angle error
        fig, ax = plt.subplots(figsize=[16,10])
        ax.set_title(f'Angle Error on 10 Test Asteroids')
        ax.set_ylabel('RMS Angle Error vs. Earth in Arcseconds')
        test_years = list(range(2000,2041))
        if run_planets:
            ax.plot(test_years, ang_err_planets, label='planets', marker='+', color='blue')
        if run_moons:
            ax.plot(test_years, ang_err_moons, label='moons', marker='x', color='green')
        if run_dwarfs:
            ax.plot(test_years, ang_err_dwarfs, label='dwarfs', marker='o', color='red')
        ax.grid()
        ax.legend()
        fig.savefig(fname=f'../figs/integration_test/planets/sim_pos_error_comp.png', bbox_inches='tight')

# ********************************************************************************************************************* 
if __name__ == '__main__':
    main()

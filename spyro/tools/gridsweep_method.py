from datetime import datetime
import spyro

def saving_source_and_receiver_location_in_csv(model):
    file_name = 'experiment/sources.txt'
    file_obj = open(file_name,'w')
    file_obj.write('Z,\tX \n')
    for source in model['acquisition']['source_pos']:
        z, x = source
        string = str(z)+',\t'+str(x)+' \n'
        file_obj.write(string)
    file_obj.close()

    file_name = 'experiment/receivers.txt'
    file_obj = open(file_name,'w')
    file_obj.write('Z,\tX \n')
    for receiver in model['acquisition']['receiver_locations']:
        z, x = receiver
        string = str(z)+',\t'+str(x)+' \n'
        file_obj.write(string)
    file_obj.close()

    return None

def executing_gridsweep(sweep_parameters):
    
    # IO parameters
    mesh_generation   = sweep_parameters['IO']['generate_meshes']
    saving_results    = sweep_parameters['IO']['output_pickle_of_wave_propagator_results']
    loading_reference = sweep_parameters['IO']['use_precomputed_reference_case']
    filenameprefix    = sweep_parameters['IO']['grid_sweep_data_filename_prefix']
    save_receiver_location = sweep_parameters['IO']['save_receiver_location']

    # Reference case for error comparison
    G_reference = sweep_parameters['reference']['G']
    P_reference = sweep_parameters['reference']['P']

    # Degrees and Gs for sweep
    Gs      = sweep_parameters['sweep_list']['DoFs']
    degrees = sweep_parameters['sweep_list']['Ps']

    # Experiment parameters
    experiment_type = sweep_parameters['experiment']['velocity_type']
    method          = sweep_parameters['experiment']['method']
    frequency       = sweep_parameters['experiment']['frequency']
    receiver_type   = sweep_parameters['experiment']['receiver_disposition']
    if experiment_type == 'homogeneous':
        minimum_mesh_velocity = sweep_parameters['experiment']['minimum_mesh_velocity']
    elif experiment_type == 'heterogeneous':
        minimum_mesh_velocity = False

    ## Generating comm
    model = spyro.tools.create_model_for_grid_point_calculation(frequency,1,method,minimum_mesh_velocity,experiment_type=experiment_type, receiver_type = receiver_type)
    comm = spyro.utils.mpi_init(model)

    ## Output file for saving data
    date = datetime.today().strftime('%Y_%m_%d')
    filename = filenameprefix+date
    text_file = open(filename+".txt", "w")
    text_file.write(experiment_type+' and '+method+' \n')

    ## Generating csv file for visualizing receiver and source position in paraview
    if save_receiver_location == True:
        saving_source_and_receiver_location_in_csv(model)

    if loading_reference == False:
        model = spyro.tools.create_model_for_grid_point_calculation(frequency, P_reference, method, minimum_mesh_velocity, experiment_type = experiment_type, receiver_type = receiver_type)
        print('Calculating reference solution of G = '+str(G_reference)+' and p = '+str(P_reference), flush = True)
        p_exact = spyro.tools.wave_solver(model, G =G_reference, comm = comm)
        spyro.io.save_shots("experiment/heterogeneous_reference_p"+str(P_reference)+"g"+str(G_reference)+"line.pck", p_exact)
    else:
        p_exact = spyro.io.load_shots("experiment/heterogeneous_reference_p"+str(P_reference)+"g"+str(G_reference)+"line.pck")

    ## Starting sweep
    for degree in degrees:
        print('\nFor p of '+str(degree), flush = True)
        text_file.write('For p of '+str(degree)+'\n')
        print('Starting sweep:', flush = True)
        ## Calculating reference solution with p=5 and g=15:
        comm.comm.barrier()
        #p_exact=spyro.io.load_shots('experiment/p_'+str(degree)+'CG_reference.pck')
        text_file.write('\tG\t\tError \n')
        for G in Gs:
            model = spyro.tools.create_model_for_grid_point_calculation(frequency, degree, method, minimum_mesh_velocity, experiment_type = experiment_type, receiver_type = receiver_type)
            print('G of '+str(G), flush = True)
            p_0 = spyro.tools.wave_solver(model, G =G, comm = comm)
            error = spyro.tools.error_calc(p_exact, p_0, model, comm = comm)
            print('With P of '+ str(degree) +' and G of '+str(G)+' Error = '+str(error), flush = True)
            text_file.write('\t'+ str(G) +'\t\t'+str(error)+' \n')
            #spyro.plots.plot_receiver_difference(model, p_exact, p_0, 1, appear=True)

    text_file.close()

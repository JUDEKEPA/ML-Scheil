%% Title: Data generation class
% Author: Zhengdi Liu (lzdseu@gmail.com)
% Last update: 2024-05-19

classdef DataGeneration < handle
    %DATAGENERATION is used for dataset generation.
    %   This class handles different alloy system raw data generation.
    %   These raw data is used to train the machine learning model.

    %   All data is organized with this format:
    %   T|Ele1|Ele2|..|EleN|Phase1|Ele1|Ele2|..|EleN|Phase2|Ele1|Ele2|..|EleN|...|
    %   This format will be automatically re-organized in python data
    %   process part.
    
    properties
        database;  % Thermo-calc database name
        elements_cell;  % The chosen alloy system
        element_num;
        temperature_range;  % Data generation temperature range
        all_phase_name;  % All phase considered
        
        data_head;
        all_data;
        
        random_composition;
        random_temperature;
    end
    
    methods
        function obj = DataGeneration(database, elements, temperature_range)
            %DATAGENERATION Construct an instance of this class
            %   Define database, elements, and temperature range here.
            obj.database = database;
            elements = upper(elements);
            obj.elements_cell = regexp(elements, ' ', 'split');
            obj.element_num = size(obj.elements_cell, 2);
            obj.temperature_range = temperature_range;
            
            data_head = [{'T'}, obj.elements_cell];
            phase_composition_data_head = [{'Phase'}, obj.elements_cell];

            % Suppose 6 phases will be generated once.
            % If any related bug occurs, increase this value.
            for i = 1:6  
                data_head = [data_head, phase_composition_data_head];
            end
            
            obj.data_head = data_head;

            tc_init_root; % Initiate the TC system
            tc_open_database(database); % Choose the database
            tc_check_error; % Check for error and reset them
            
            tc_element_select(elements);
            tc_get_data;
            
            [~, all_phase_name] = tc_list_phase();
            % Find indices of cells containing '#'
            containsHash = contains(all_phase_name, '#');

            % Remove cells containing '#'
            obj.all_phase_name = transpose(all_phase_name(~containsHash));
            
            tc_set_condition('p', 101325);
            tc_set_condition('n', 1);
            
        end
        
        function dirichlet_random(obj, data_num)
            %DIRICHLET_RANDOM generate random composition in simplex
            %composition space.

            %   Dirichlet distribution ensures the random distribution
            %   while the sum of the compositon is always 1.
            
            % Number of dimensions
            n = obj.element_num; % For a 5-dimensional simplex

            % Preallocate the output matrix
            samples = zeros(data_num, n);

            for i = 1:data_num
                % Generate n random exponential(1) variables
                random_exps = exprnd(1, [1, n]);

                % Normalize them so that their sum is 1
                samples(i, :) = random_exps / sum(random_exps);
            end
            
            obj.random_composition = samples;
        end
        
        function generate_random_t(obj, data_num)
            %GENERATE_RANDOM_T is used for generate random temperature.

            % Preallocate the output matrix
            samples = zeros(data_num, 1);

            % random ration between temperature range
            ratio = rand(data_num, 1); 
            
            for i = 1:data_num
                samples(i) = obj.temperature_range(1) + ratio(i) * ...
                (obj.temperature_range(2) - obj.temperature_range(1));
            end
            
            obj.random_temperature = samples;
        end
        
        function result = single_data_calculation(obj, composition, temperature)
            %SINGLE_DATA_CALCULATION is used to get single calculation
            %result.

            %   result is organized into the format described in class
            %   description.

            result = cell(1, size(obj.data_head, 1));
            result(1) = num2cell(temperature);
            result(2:obj.element_num+1) = num2cell(composition);
            
            % Condition setting
            tc_set_condition('t', temperature);
            
            for i = 1:obj.element_num-1
                tc_set_condition(['x(', cell2mat(obj.elements_cell(i)), ')'],...
                composition(i));
            end
            
            tc_compute_equilibrium;
            
            % Preallocate the output matrix
            generated_phase_frac = cell(1, size(obj.all_phase_name, 2));
            generated_phase_frac(:) = {0};
            
            comp_cell = cell(1, size(obj.all_phase_name, 2)*obj.element_num);
            comp_cell(:) = {0};
            
            [phase_amount, all_phase_name] = tc_list_phase();
            phase_frac=zeros(phase_amount,1);

            % Get generated phase fraction
            i = 1;
            while(i<phase_amount+1)
                phase_frac(i) = obj.get_phase_frac(all_phase_name(i));
                i = i+1;
            end
            
            i=1;
            [m, ~] = size(all_phase_name);

            % Spinodal Decomposition is ignored.
            while(i<m+1)
                if phase_frac(i)>0
                    generated_phase = cell2mat(all_phase_name(i));
                    
                    if contains(generated_phase, '#')
                        generated_phase_nowell = generated_phase(1:strfind(temp, '#')-1);
                    else
                        generated_phase_nowell = generated_phase;
                    end
                    
                    % Phase fraction part
                    phase_index = find(strcmp(obj.all_phase_name, generated_phase_nowell));
                    former_frac = cell2mat(generated_phase_frac(phase_index));
                    update_frac = former_frac + phase_frac(i);
                    generated_phase_frac(phase_index) = num2cell(update_frac);
                    
                    % Phase composition part
                    start_index = (phase_index-1) * obj.element_num + 1;
                    former_comp = cell2mat(comp_cell(start_index:start_index + obj.element_num - 1));
                    weighted_former = former_comp * former_frac;
                    
                    % Get the composition
                    single_pha_comp = [];
                    for j = 1:obj.element_num
                        format_str = ['x(', generated_phase, ',', cell2mat(obj.elements_cell(j)), ')'];
                        comp = tc_get_value(format_str);
                        single_pha_comp = [single_pha_comp, comp];
                    end

                    % Deal with '#' and possible spinodal decomposition
                    weighted_temp = phase_frac(i)*single_pha_comp;
                    update_comp = (weighted_former + weighted_temp)/(former_frac + phase_frac(i));
                    comp_cell(start_index:start_index + obj.element_num - 1) = num2cell(update_comp);
                    
                end
                i=i+1;
            end
            
            % Organize data into defined format
            generated_phase_count = 0;
            for i = 1:size(generated_phase_frac, 2)
                if generated_phase_frac{i} > 0
                    start_index = (i - 1) * obj.element_num + 1;
                    phase_comp = comp_cell(start_index:start_index + obj.element_num - 1);
                    
                    start_index = generated_phase_count * (obj.element_num + 1) + (obj.element_num + 2);
                    result(start_index) = obj.all_phase_name(i);
                    result(start_index+1:start_index+obj.element_num) = phase_comp;
                    
                    generated_phase_count = generated_phase_count + 1;
                end
            end 
        end
        
        function data_generate(obj, data_num)
            %DATA_GENERATION is the external API for data generation.
            % data_num is the number of data that need to generate.
            obj.all_data = cell(data_num, size(obj.data_head, 2));
            obj.dirichlet_random(data_num);
            obj.generate_random_t(data_num);
            
            for i = 1:data_num
                result = obj.single_data_calculation(obj.random_composition(i, :), ...
                                                     obj.random_temperature(i));
                obj.all_data(i, 1:size(result, 2)) = result;
                disp(i);
            end
            
            obj.all_data = [obj.data_head;obj.all_data];
        end
        
        function save_in_xlsx(obj)
            random_idx = randi(100, 1); % In case former data be overwritten.
            file_name = '';
            for i = 1:size(obj.elements_cell, 2)
                file_name = [file_name, obj.elements_cell{i}];
            end
            file_name = [file_name, char(string(random_idx)), '.xlsx'];
            xlswrite(file_name, obj.all_data);
        end
    end
    
    methods(Static)
        function phase_frac = get_phase_frac(phase_name)
            %GET_PHASE_FRAC
            name_modified = ['np(', char(phase_name), ')'];
            phase_frac = tc_get_value(name_modified);
        end
    end
    
end


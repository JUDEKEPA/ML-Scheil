scheil_calc_fenicrmn = ScheilCalculation('tchea5', 'fe ni cr mn', [0.25, 0.25, 0.25, 0.25]);
scheil_calc_fenicrmn.calculate();
%scheil_calc_fenicrmn.integrate_storage('result path');
scheil_calc_fenicrmn.draw_scheil_curve();
scheil_calc_fenicrmn.draw_comp_change_in_phase('LIQUID', {'fe' 'ni' 'cr' 'mn'});


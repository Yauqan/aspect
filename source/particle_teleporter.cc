
/*
  Copyright (C) 2011 - 2025 by the authors of the ASPECT code.

  This file is part of ASPECT.

  ASPECT is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2, or (at your option)
  any later version.

  ASPECT is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with ASPECT; see the file LICENSE.  If not see
  <http://www.gnu.org/licenses/>.
*/

#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/parsed_function.h>
#include <deal.II/fe/fe_values.h>
#include <aspect/global.h>
#include <aspect/utilities.h>
#include <aspect/simulator_signals.h>
#include <aspect/parameters.h>

// Maybe need something from this?
#include <aspect/geometry_model/interface.h>
#include <aspect/particle/manager.h>
#include <aspect/simulator_access.h>
#include <aspect/prescribed_dilation/interface.h>
#include <aspect/material_model/interface.h>


namespace aspect
{
  // Global variables (to be set by parameters)
  bool teleport_particles;
  int number_of_jumps;

  // Because we do not initially know what dimension we're in, we need
  // function parser objects for both 2d and 3d.
  Functions::ParsedFunction<2> generation_indicator_function (1);
  Functions::ParsedFunction<2> prescribed_velocity_function (2);



  /**
   * Declare additional parameters.
   */
  void declare_parameters(const unsigned int dim,
                          ParameterHandler &prm)
  {
    prm.declare_entry ("Teleport particles", "false",
                       Patterns::Bool (),
                       "Generates random particles and teleports them"
                      );

    prm.enter_subsection ("Particle teleportation");
    {
      prm.declare_entry ("Number of jumps", "100",
                          Patterns::Integer (),
                          "Defines how many jumps a particle should do");
      prm.enter_subsection ("Indicator function");
      {
        Functions::ParsedFunction<2>::declare_parameters (prm, 1);
      }
      prm.leave_subsection ();

      prm.enter_subsection ("Velocity function");
      {
        Functions::ParsedFunction<2>::declare_parameters (prm, 2);
      }
      prm.leave_subsection ();
    }
    prm.leave_subsection ();
  }



  template <int dim>
  void parse_parameters(const Parameters<dim> &,
                        ParameterHandler &prm)
  {
    teleport_particles = prm.get_bool ("Teleport particles");
    if ( teleport_particles )
      AssertThrow ( dim == 2, ExcMessage ( "Sorry, didn't implement teleportation for 3D..." ) );
    prm.enter_subsection ("Particle teleportation");
    {
      number_of_jumps = prm.get_integer ("Number of jumps");
      prm.enter_subsection("Indicator function");
      {
        try
          {
            generation_indicator_function.parse_parameters (prm);
          }
        catch (...)
          {
            std::cerr << "ERROR: FunctionParser failed to parse\n"
                      << "\t'Particle teleportation.Indicator function'\n"
                      << "with expression\n"
                      << "\t'" << prm.get("Function expression") << "'";
            throw;
          }
      }
      prm.leave_subsection();

      prm.enter_subsection("Velocity function");
      {
        try
          {
            prescribed_velocity_function.parse_parameters (prm);
          }
        catch (...)
          {
            std::cerr << "ERROR: FunctionParser failed to parse\n"
                      << "\t'Particle teleportation.Velocity function'\n"
                      << "with expression\n"
                      << "\t'" << prm.get("Function expression") << "'";
            throw;
          }
      }
      prm.leave_subsection();
    }
    prm.leave_subsection ();
  }

  void do_particle_stuff ( const SimulatorAccess<2> &SA )
  {
    if ( teleport_particles ) {
      // Declare fe_values to be able to loop over cells
      const Quadrature<2> & quadrature_formula = SA.introspection().quadratures.temperature;
      const unsigned int n_q_points = quadrature_formula.size();
      const Mapping<2> & mapping = SA.get_mapping();
      FEValues<2> fe_values ( mapping,
                                SA.get_fe(),
                                quadrature_formula,
                                update_values |
                                update_gradients |
                                update_quadrature_points |
                                update_JxW_values);
      
      // Declare everything needed for particles here
      Particle::ParticleHandler<2> particle_handler;
      particle_handler.initialize (SA.get_triangulation(), mapping);

      // Somehow generate particles
      MaterialModel::MaterialModelInputs<2> in(fe_values.n_quadrature_points, SA.n_compositional_fields());
      for (const auto &cell : SA.get_dof_handler().active_cell_iterators())
        if (cell->is_locally_owned())
        {
          fe_values.reinit (cell);
          in.reinit(fe_values, cell, SA.introspection(), SA.get_solution());
          const Particles::internal::LevelInd cell_info = Particles::internal::LevelInd (cell->level(), cell->index());
          for (unsigned int q=0; q<n_q_points; ++q)
          {
            if ( generation_indicator_function.value ( in.position[q] ) > 0.5 )
            {
              // Create particle
              types::particle_index local_next_particle_index = particle_handler.get_next_free_particle_index();
              typename Particle::ParticleHandler<2>::particle_iterator new_particle = 
                particle_handler.insert_particle( Particles::Particle<2> ( in.position[q], mapping.transform_real_to_unit_cell ( cell, in.position[q] ), local_next_particle_index ),
                                                  typename parallel::distributed::Triangulation<2>::cell_iterator (&SA.get_triangulation(),
                                                      cell_info.first,
                                                      cell_info.second));
            }
          }
        }
      
      // Move particles, THIS IS WHAT IS INTERESTING FOR YOU
      for ( int j = 0; j < number_of_jumps; ++j )
      {
        for ( auto & cell: SA.get_dof_handler().active_cell_iterators() )                                        // 
          if ( cell->is_locally_owned() )                                                                           //  
            if ( particle_handler.n_particles_in_cell(cell) > 0 )                                                   //   Here we iterate over all particles
            {                                                                                                       //  
              for ( typename Particles::ParticleAccessor<2> particle : particle_handler.particles_in_cell(cell) )   // 
              {
                Point<2> & location = particle.get_location();
                Vector<double> movevector;
                prescribed_velocity_function.vector_value ( location, movevector );
                if ( movevector.l2_norm() > cell->diameter()/2.0 )                       // We dont want to end up too far away in completely different cell...
                  for ( int k = 0; k < 2; ++k )  
                    movevector[k] = movevector[k]/movevector.l2_norm()*cell->diameter()/2.0;     //
                for ( int k = 0; k < 2; ++k )
                  location[k] += movevector[k];          // Here we move the particle we found in the direction we want
              }
            }
        particle_handler.sort_particles_into_subdomains_and_cells();    // This is crutial, this redistributes particles between cells and processors
      }

      // Clean up, destroy particles
      for ( auto &cell: SA.get_dof_handler().active_cell_iterators() )
        if ( cell->is_locally_owned() )
        {
          if ( particle_handler.n_particles_in_cell(cell) > 0 )
          {
            while ( particle_handler.n_particles_in_cell(cell) > 0 )
            {
              particle_handler.remove_particle(particle_handler.particles_in_cell(cell).begin());
            }
          }
        }
    }
    return;
  }

  // Connect declare_parameters and parse_parameters to appropriate signals.
  void parameter_connector ()
  {
    SimulatorSignals<2>::declare_additional_parameters.connect (&declare_parameters);
    SimulatorSignals<3>::declare_additional_parameters.connect (&declare_parameters);

    SimulatorSignals<2>::parse_additional_parameters.connect (&parse_parameters<2>);
    SimulatorSignals<3>::parse_additional_parameters.connect (&parse_parameters<3>);
  }

  // Connect constraints function to correct signal.
  template <int dim>
  void signal_connector (SimulatorSignals<dim> &signals)
  {
    if constexpr ( dim == 2 )
      signals.pre_nonlinear_solver.connect (&do_particle_stuff);
  }

  // Tell ASPECT to send signals to the connector functions
  ASPECT_REGISTER_SIGNALS_PARAMETER_CONNECTOR(parameter_connector)
  ASPECT_REGISTER_SIGNALS_CONNECTOR(signal_connector<2>, signal_connector<3>)
}
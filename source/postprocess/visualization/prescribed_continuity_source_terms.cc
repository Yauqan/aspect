/*
  Copyright (C) 2011 - 2024 by the authors of the ASPECT code.

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


#include <aspect/postprocess/visualization/prescribed_continuity_source_terms.h>
#include <aspect/geometry_model/interface.h>



namespace aspect
{
  namespace Postprocess
  {
    namespace VisualizationPostprocessors
    {
      template <int dim>
      PrescribedContinuitySourceTerms<dim>::
      PrescribedContinuitySourceTerms ()
        :
        DataPostprocessorScalar<dim> ("prescribed_continuity_source_terms",
                                      update_quadrature_points),
        Interface<dim>("kg/m^3/s")
      {}



      template <int dim>
      void
      PrescribedContinuitySourceTerms<dim>::
      evaluate_vector_field(const DataPostprocessorInputs::Vector<dim> &input_data,
                            std::vector<Vector<double>> &computed_quantities) const
      {
        const unsigned int n_quadrature_points = input_data.solution_values.size();
        Assert (computed_quantities.size() == n_quadrature_points,    ExcInternalError());
        Assert (computed_quantities[0].size() == 1,                   ExcInternalError());
        Assert (input_data.solution_values[0].size() == this->introspection().n_components,           ExcInternalError());

        Functions::ParsedFunction<dim> & function = this->get_parameters().prescribed_continuity_source_terms_function;
        const Utilities::Coordinates::CoordinateSystem & coordinate_system = this->get_parameters().prescribed_continuity_source_terms_coordinate_system;
        const double time = this->get_time();

        if (this->convert_output_to_years())
          function.set_time (time / year_in_seconds);
        else
          function.set_time (time);
        
        for (unsigned int q=0; q<n_quadrature_points; ++q)
          {
            const Utilities::NaturalCoordinate<dim> point =
              this->get_geometry_model().cartesian_to_other_coordinates(input_data.evaluation_points[q], coordinate_system);
            computed_quantities[q](0) = function.value(Utilities::convert_array_to_point<dim>(point.get_coordinates()));
          }
      }
    }
  }
}


// explicit instantiations
namespace aspect
{
  namespace Postprocess
  {
    namespace VisualizationPostprocessors
    {
      ASPECT_REGISTER_VISUALIZATION_POSTPROCESSOR(PrescribedContinuitySourceTerms,
                                                  "prescribed continuity source terms",
                                                  "A visualization output postprocessor that outputs "
                                                  "the prescribed continuity source terms, which are"
                                                  "determined by the user."
                                                  "\n\n"
                                                  "Physical units: $\\frac{text{kg}}{\\text{m}^3\\text{s}}$.")
    }
  }
}

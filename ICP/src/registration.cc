#include "registration_problem.h"

using RegistrationProblem = registration::RegistrationProblem;
constexpr int KMaxIteration = 10;

int main() {

  RegistrationProblem ICPproblem;

  // Setup the problem by giving it a source point clould
  ICPproblem.Setup("/Users/estellejiao/Desktop/LearningSLAM/src/SLAM_learning/ICP/test/bun01.pcd");

  // Visualise the target and the initial source points
  // Press Q to quit the visulisation
  ICPproblem.Visulise();

  // Start the registration
  // We perform associate and solve for KMaxIteration times

  while (ICPproblem.num_iteration_ < KMaxIteration) {
    std::cout << "Iteration: " << ICPproblem.num_iteration_ << std::endl;

    // 1. we find the association points between source and target points
    ICPproblem.FindAssociation();

    // 2. we solve the minimisation problem via ceres and update the source points
    ICPproblem.SolveAndUpdate();

    // 3. we visualise for each step
    // ICPproblem.Visulise();
  }

  // Visualise the final result
  ICPproblem.Visulise();

  return 0;
}
#include <iostream>
#include <iomanip>

// #include <unitree/robot/channel/channel_publisher.hpp>
#include <unitree/idl/go2/LowState_.hpp>
#include <unitree/robot/channel/channel_subscriber.hpp>
// #include <unitree/idl/go2/LowCmd_.hpp>
#include <unitree/idl/go2/WirelessController_.hpp>
#include <unitree/robot/client/client.hpp>
#include <unitree/common/thread/thread.hpp>
#include <unitree/common/time/time_tool.hpp>
#include <unitree/robot/go2/robot_state/robot_state_client.hpp>

#define TOPIC_LOWSTATE "rt/lowstate"

struct motor_states
{
    float q[12];
    float dq[12];
    float ddq[12];
    float tau_est[12];
} motorStates;

struct imu_states
{
    float quat[4];
    float rpy[3];
    float omegaBody[3];
    float acceBody[3];
} IMUStates;

float contact_force[4];

void LowStateMessageHandler(const void *message)
{
    unitree_go::msg::dds_::LowState_ low_state{};
    low_state = *(unitree_go::msg::dds_::LowState_ *)message;

    for (int i = 0; i < 12; i++)
    {
        motorStates.q[i] = low_state.motor_state()[i].q();
        motorStates.dq[i] = low_state.motor_state()[i].dq();
        motorStates.ddq[i] = low_state.motor_state()[i].ddq();
        motorStates.tau_est[i] = low_state.motor_state()[i].tau_est();
    }

    for (int i = 0; i < 4; i++)
    {
        IMUStates.quat[i] = low_state.imu_state().quaternion()[i];
    }
    for (int i = 0; i < 3; i++)
    {
        IMUStates.rpy[i] = low_state.imu_state().rpy()[i];
        IMUStates.omegaBody[i] = low_state.imu_state().gyroscope()[i];
        IMUStates.acceBody[i] = low_state.imu_state().accelerometer()[i];
    }

    for (int i = 0; i < 4; i++)
    {
        contact_force[i] = low_state.foot_force()[i];
    }

    std::cout << "=============================================================================" << std::endl
              << std::setw(28) << "Go2 LowState Messages" << std::endl
              << "=============================================================================" << std::endl
              << "Motor ID:\t" << "Angle(rad)\t" << "Velocity\t" << "Acceleration\t" << "Torque" << std::endl
              << std::left
              << std::setw(4) << "0" << std::setw(16) << motorStates.q[0] << "\t" << motorStates.dq[0] << "\t" << motorStates.ddq[0] << "\t" << motorStates.tau_est[0] << std::endl
              << std::setw(4) << "1" << std::setw(16) << motorStates.q[1] << "\t" << motorStates.dq[1] << "\t" << motorStates.ddq[1] << "\t" << motorStates.tau_est[1] << std::endl
              << std::setw(4) << "2" << std::setw(16) << motorStates.q[2] << "\t" << motorStates.dq[2] << "\t" << motorStates.ddq[2] << "\t" << motorStates.tau_est[2] << std::endl
              << std::setw(4) << "3" << std::setw(16) << motorStates.q[3] << "\t" << motorStates.dq[3] << "\t" << motorStates.ddq[3] << "\t" << motorStates.tau_est[3] << std::endl
              << std::setw(4) << "4" << std::setw(16) << motorStates.q[4] << "\t" << motorStates.dq[4] << "\t" << motorStates.ddq[4] << "\t" << motorStates.tau_est[4] << std::endl
              << std::setw(4) << "5" << std::setw(16) << motorStates.q[5] << "\t" << motorStates.dq[5] << "\t" << motorStates.ddq[5] << "\t" << motorStates.tau_est[5] << std::endl
              << std::setw(4) << "6" << std::setw(16) << motorStates.q[6] << "\t" << motorStates.dq[6] << "\t" << motorStates.ddq[6] << "\t" << motorStates.tau_est[6] << std::endl
              << std::setw(4) << "7" << std::setw(16) << motorStates.q[7] << "\t" << motorStates.dq[7] << "\t" << motorStates.ddq[7] << "\t" << motorStates.tau_est[7] << std::endl
              << std::setw(4) << "8" << std::setw(16) << motorStates.q[8] << "\t" << motorStates.dq[8] << "\t" << motorStates.ddq[8] << "\t" << motorStates.tau_est[8] << std::endl
              << std::setw(4) << "9" << std::setw(16) << motorStates.q[9] << "\t" << motorStates.dq[9] << "\t" << motorStates.ddq[9] << "\t" << motorStates.tau_est[9] << std::endl
              << std::setw(4) << "10" << std::setw(16) << motorStates.q[10] << "\t" << motorStates.dq[10] << "\t" << motorStates.ddq[10] << "\t" << motorStates.tau_est[10] << std::endl
              << std::setw(4) << "11" << std::setw(16) << motorStates.q[11] << "\t" << motorStates.dq[11] << "\t" << motorStates.ddq[11] << "\t" << motorStates.tau_est[11] << std::endl
              << "-----------------------------------------------------------------------------" << std::endl
              << "Quaternion:\t" << "w: " << IMUStates.quat[0] << "\t" << "x: " << IMUStates.quat[1] << "\t" << "y: " << IMUStates.quat[2] << "\t" << "z: " << IMUStates.quat[3] << std::endl
              << "Euler Angle:\t" << "roll: " << IMUStates.rpy[0] << "\t" << "pitch: " << IMUStates.rpy[1] << "\t" << "yaw: " << IMUStates.rpy[2] << std::endl
              << "Omega Body:\t" << "x: " << IMUStates.omegaBody[0] << "\t" << "y: " << IMUStates.omegaBody[1] << "\t" << "z: " << IMUStates.omegaBody[2] << std::endl
              << "Acceleration Body:\t" << "x: " << IMUStates.acceBody[0] << "\t" << "y: " << IMUStates.acceBody[1] << "\t" << "z: " << IMUStates.acceBody[2] << std::endl
              << "Foot Force:\t" << "FR: " << contact_force[0] << "\t" << "FL: " << contact_force[1] << "\t" << "RR: " << contact_force[2] << "\t" << "RL: " << contact_force[3] << std::endl
              << "=============================================================================" << std::endl;
}

int main(int argc, char **argv)
{
    if (argc < 2)
    {
        std::cout << "Usage: ./" << argv[0] << " networkInterface" << std::endl;
        exit(1);
    }

    unitree::robot::ChannelFactory::Instance()->Init(0, argv[1]);

    // unitree_go::msg::dds_::LowState_ low_state{};
    unitree::robot::ChannelSubscriberPtr<unitree_go::msg::dds_::LowState_> lowstate_subscriber;
    lowstate_subscriber.reset(new unitree::robot::ChannelSubscriber<unitree_go::msg::dds_::LowState_>(TOPIC_LOWSTATE));
    lowstate_subscriber->InitChannel(std::bind(&LowStateMessageHandler, std::placeholders::_1), 1);

    while (true)
    {
        sleep(10);
    }
    return 0;
}
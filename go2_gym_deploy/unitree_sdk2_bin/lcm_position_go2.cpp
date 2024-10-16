""
    "
    lcm_position_go2.cpp主要负责与 Go2 机器人交互，获取机器人状态信息，并将控制指令发送给机器人。
        这里不涉及网络模型的加载和推理，这部分由 deploy_policy.py 实现。
    ""
    "

// lcm related headfile
#include <lcm/lcm-cpp.hpp>
#include "leg_control_data_lcmt.hpp"
#include "state_estimator_lcmt.hpp"
#include "rc_command_lcmt.hpp"
#include "pd_tau_targets_lcmt.hpp"
// standard headfile
#include <iostream>
#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include <cmath>
// unitree_sdk2 related headfile
#include <unitree/robot/channel/channel_publisher.hpp>
#include <unitree/robot/channel/channel_subscriber.hpp>
#include <unitree/idl/go2/LowState_.hpp>
#include <unitree/idl/go2/LowCmd_.hpp>
#include <unitree/idl/go2/WirelessController_.hpp>
#include <unitree/robot/client/client.hpp>
#include <unitree/common/thread/thread.hpp>
#include <unitree/common/time/time_tool.hpp>
#include <unitree/robot/go2/robot_state/robot_state_client.hpp>

#define TOPIC_LOWCMD "rt/lowcmd"
#define TOPIC_LOWSTATE "rt/lowstate"
#define TOPIC_JOYSTICK "rt/wirelesscontroller"

    // 为保证项目代码的稳定性和易理解，没有采用unitree_sdk2中采用的using namespace语句

    constexpr double PosStopF = (2.146E+9f);
constexpr double VelStopF = (16000.0f);

// 无需更改：Unitree 提供的电机校验函数
uint32_t crc32_core(uint32_t *ptr, uint32_t len) // 阅读完成
{
    unsigned int xbit = 0;
    unsigned int data = 0;
    unsigned int CRC32 = 0xFFFFFFFF;
    const unsigned int dwPolynomial = 0x04c11db7;

    for (unsigned int i = 0; i < len; i++)
    {
        xbit = 1 << 31;
        data = ptr[i];
        for (unsigned int bits = 0; bits < 32; bits++)
        {
            if (CRC32 & 0x80000000)
            {
                CRC32 <<= 1;
                CRC32 ^= dwPolynomial;
            }
            else
            {
                CRC32 <<= 1;
            }

            if (data & xbit)
                CRC32 ^= dwPolynomial;
            xbit >>= 1;
        }
    }

    return CRC32;
}

// 遥控器键值联合体，摘自unitree_sdk2，无需更改
typedef union // 阅读完成
{
    struct
    {
        uint8_t R1 : 1;
        uint8_t L1 : 1;
        uint8_t start : 1;
        uint8_t select : 1;
        uint8_t R2 : 1;
        uint8_t L2 : 1;
        uint8_t F1 : 1;
        uint8_t F2 : 1;
        uint8_t A : 1;
        uint8_t B : 1;
        uint8_t X : 1;
        uint8_t Y : 1;
        uint8_t up : 1;
        uint8_t right : 1;
        uint8_t down : 1;
        uint8_t left : 1;
    } components;
    uint16_t value;
} xKeySwitchUnion;

class Custom
{
public:
    explicit Custom() {} // √
    ~Custom() {}         // √

    void Init();                                                         // √
    void InitLowCmd();                                                   // √
    void Loop();                                                         // √
    void LowStateMessageHandler(const void *messages);                   // √
    void JoystickHandler(const void *message);                           // √
    void InitRobotStateClient();                                         // √
    void SwitchService(const std::string &serviceName, uint32_t status); // √

    void lcm_send();                                                                                                   // √
    void lcm_receive();                                                                                                // √
    void lcm_receive_Handler(const lcm::ReceiveBuffer *rbuf, const std::string &chan, const pd_tau_targets_lcmt *msg); // √
    void LowCmdWrite();                                                                                                // √

    void SetNominalPose();                                       // √
    uint32_t QueryServiceStatus(const std::string &serviceName); // √

    //------------------------------------------------
    leg_control_data_lcmt leg_control_lcm_data = {0};
    state_estimator_lcmt body_state_simple = {0};
    pd_tau_targets_lcmt joint_command_simple = {0};
    rc_command_lcmt rc_command = {0};
    //------------------------------------------------

    unitree_go::msg::dds_::LowState_ low_state{};          // √
    unitree_go::msg::dds_::LowCmd_ low_cmd{};              // √
    unitree_go::msg::dds_::WirelessController_ joystick{}; // √
    unitree::robot::go2::RobotStateClient rsc;             // √

    unitree::robot::ChannelPublisherPtr<unitree_go::msg::dds_::LowCmd_> lowcmd_publisher;            // √
    unitree::robot::ChannelSubscriberPtr<unitree_go::msg::dds_::LowState_> lowstate_subscriber;      // √
    unitree::robot::ChannelSubscriberPtr<unitree_go::msg::dds_::WirelessController_> joystick_suber; // √
    lcm::LCM lc;                                                                                     // √

    xKeySwitchUnion key; // √
    int mode = 0;        // 某种模式，可能指的是运动模式
    int motiontime = 0;  // 底层控制指令的下发次数 √
    float dt = 0.002;    // unit [second] 控制步长 √
    bool firstRun;       // √

    /*LowCmd write thread*/
    // DDS相关的底层命令发送线程指针
    unitree::common::ThreadPtr LcmSendThreadPtr;     // √
    unitree::common::ThreadPtr LcmRecevThreadPtr;    // √
    unitree::common::ThreadPtr lowCmdWriteThreadPtr; // √
};

// 初始化 RobotStateClient 类的实例 √
void Custom::InitRobotStateClient() // 阅读完成
{
    rsc.SetTimeout(10.0f); // 设置超时时间
    rsc.Init();            // 初始化rsc
}

// 获取主运控服务（sport_mode）的当前状态 √
uint32_t Custom::QueryServiceStatus(const std::string &serviceName) // 阅读完成
{
    std::vector<unitree::robot::go2::ServiceState> serviceStateList;
    uint32_t serviceStatus;

    rsc.ServiceList(serviceStateList);

    size_t count = serviceStateList.size();
    for (uint32_t i = 0; i < count; i++)
    {
        const unitree::robot::go2::ServiceState &serviceState = serviceStateList[i];
        if (serviceState.name == serviceName)
        {
            if (serviceState.status == 0)
            {
                std::cout << "Service: " << serviceState.name << " is active." << std::endl;
                serviceStatus = 1;
            }
            else
            {
                std::cout << "Service:" << serviceState.name << " is inactive." << std::endl;
                serviceStatus = 0;
            }
        }
    }
    return serviceStatus;
}

// 关闭主运控服务（sport_mode） √
void Custom::SwitchService(const std::string &serviceName, uint32_t status) // 阅读完成
{
    rsc.ServiceSwitch(serviceName, status);
}

// 读取底层状态 √
void Custom::LowStateMessageHandler(const void *message) // 阅读完成
{
    // 用sdk2读取的底层state
    low_state = *(unitree_go::msg::dds_::LowState_ *)message;
}

// 保存遥控器消息 √
void Custom::JoystickHandler(const void *message) // 阅读完成
{
    // 遥控器信号
    joystick = *(unitree_go::msg::dds_::WirelessController_ *)message;
    key.value = joystick.keys();
}

// -------------------------------------------------------------------------------
// 线程 1 ： lcm send 线程
// 此线程作用：实时通过unitree_sdk2读取low_state信号和joystick信号，并发送给lcm中间件
void Custom::lcm_send() // 阅读完成 √
{
    // leg_control_lcm_data
    for (int i = 0; i < 12; i++)
    {
        leg_control_lcm_data.q[i] = low_state.motor_state()[i].q();
        leg_control_lcm_data.qd[i] = low_state.motor_state()[i].dq();
        leg_control_lcm_data.tau_est[i] = low_state.motor_state()[i].tau_est();
    }
    // 从IMU读取姿态信息
    for (int i = 0; i < 4; i++)
    {
        // 四元数
        body_state_simple.quat[i] = low_state.imu_state().quaternion()[i];
    }
    for (int i = 0; i < 3; i++)
    {
        // roll pitch yaw
        body_state_simple.rpy[i] = low_state.imu_state().rpy()[i];
        // 三轴线性加速度
        body_state_simple.aBody[i] = low_state.imu_state().accelerometer()[i];
        // 三轴角速度
        body_state_simple.omegaBody[i] = low_state.imu_state().gyroscope()[i];
    }
    for (int i = 0; i < 4; i++)
    {
        // 足端触地力
        body_state_simple.contact_estimate[i] = low_state.foot_force()[i];
    }
    // 遥控器摇杆数值和按键值
    rc_command.left_stick[0] = joystick.lx();
    rc_command.left_stick[1] = joystick.ly();
    rc_command.right_stick[0] = joystick.rx();
    rc_command.right_stick[1] = joystick.ry();
    rc_command.right_lower_right_switch = key.components.R2;
    rc_command.right_upper_switch = key.components.R1;
    rc_command.left_lower_left_switch = key.components.L2;
    rc_command.left_upper_switch = key.components.L1;

    if (key.components.A > 0)
    {
        mode = 0;
    }
    else if (key.components.B > 0)
    {
        mode = 1;
    }
    else if (key.components.X > 0)
    {
        mode = 2;
    }
    else if (key.components.Y > 0)
    {
        mode = 3;
    }
    else if (key.components.up > 0)
    {
        mode = 4;
    }
    else if (key.components.right > 0)
    {
        mode = 5;
    }
    else if (key.components.down > 0)
    {
        mode = 6;
    }
    else if (key.components.left > 0)
    {
        mode = 7;
    }

    rc_command.mode = mode;

    lc.publish("leg_control_data", &leg_control_lcm_data);
    // 发布机器人的状态信息到LCM中间件
    lc.publish("state_estimator_data", &body_state_simple);
    // 发步机器人姿态信息到LCM中间件
    lc.publish("rc_command", &rc_command);
    // 发步遥控器信号到LCM中间件

    // std::cout << "loop: messsages are sending ......" << std::endl;
}

// -------------------------------------------------------------------------------
// 线程 2 ： lcm receive 线程
// 此线程作用：实时通过lcm中间件读取pytorch神经网络输出的期望关节控制信号（q, qd, kp, kd, tau_ff）
// 查看 go2_gym_deploy/envs/lcm_agent.py 文件，可以知道：
// 神经网络只输出期望的q，而kp，kd是可以自定义设置的, qd 和 tau_ff 被设置为0
void Custom::lcm_receive_Handler(const lcm::ReceiveBuffer *rbuf, const std::string &chan, const pd_tau_targets_lcmt *msg) // 阅读完成 √
{
    (void)rbuf;                  // 声明为未使用的参数
    (void)chan;                  // 声明为未使用的参数
    joint_command_simple = *msg; // 接收神经网络输出的关节信号
}

// 此处参考lcm推荐的标准格式，循环处理，接受lcm消息
void Custom::lcm_receive() // 阅读完成 √
{
    while (true)
    {
        lc.handle(); // 阻塞并等待传入的消息
    }
}

// -------------------------------------------------------------------------------
// 线程 3 ： unitree_sdk2 command write 线程
// 此线程作用：初始化low_cmd，经过合理的状态机后，电机将执行神经网络的输出

// InitLowCmd()该函数用于初始化，设置 LowCmd 类型的 low_cmd 结构体。该函数放置于 Custom 类的构造函数运行一次即可。
void Custom::InitLowCmd() // 阅读完成 √
{
    /* LowCmd 类型中的 head 成员 表示帧头，此帧头将用于 CRC 校验。head、levelFlag、gpio 等按例程所示设置为默认值即可。*/
    low_cmd.head()[0] = 0xFE;
    low_cmd.head()[1] = 0xEF;
    low_cmd.level_flag() = 0xFF;
    low_cmd.gpio() = 0;

    /*LowCmd 类型中有 20 个 motor_cmd 成员，
    每一个成员的命令用于控制 Go2 机器人上相对应的一个电机，
    但 Go2 机器人上只有 12 个电机，
    故仅有前 12 个有效，剩余的8个起保留作用。*/
    for (int i = 0; i < 20; i++)
    {
        /*此行命令中将 motor_cmd 成员的 mode 变量设置为 0x01，
        0x01 表示将电机设置为伺服模式。
        如果用户在调试过程中发现无法控制 Go2 机器人的关节电机，
        请检查变量的值是否为0x01。*/
        low_cmd.motor_cmd()[i].mode() = (0x01); // motor switch to servo (PMSM) mode
        low_cmd.motor_cmd()[i].q() = (PosStopF);
        low_cmd.motor_cmd()[i].dq() = (VelStopF);
        low_cmd.motor_cmd()[i].kp() = (0);
        low_cmd.motor_cmd()[i].kd() = (0);
        low_cmd.motor_cmd()[i].tau() = (0);
    }
}

// SetNominalPose()函数用于设置趴下时初始的关节角度
void Custom::SetNominalPose() // 阅读完成 √
{
    // 运行此cpp文件后，不仅是初始化通信
    // 同样会在趴下时的初始化关节角度
    // 将各个电机都设置为位置模式
    for (int i = 0; i < 12; i++)
    {
        joint_command_simple.qd_des[i] = 0;
        joint_command_simple.tau_ff[i] = 0;
        joint_command_simple.kp[i] = 20;  // kp = 20
        joint_command_simple.kd[i] = 0.5; // kd = 0.5
    }

    // 趴下时的关节角度
    joint_command_simple.q_des[0] = -0.3;    // 左前机身关节    右前机身关节
    joint_command_simple.q_des[1] = 1.2;     // 左前大腿关节    右前大腿关节
    joint_command_simple.q_des[2] = -2.721;  // 左前小腿关节    右前小腿关节
    joint_command_simple.q_des[3] = 0.3;     // 右前机身关节    左前机身关节
    joint_command_simple.q_des[4] = 1.2;     // 右前大腿关节    左前大腿关节
    joint_command_simple.q_des[5] = -2.721;  // 右前小腿关节    左前小腿关节
    joint_command_simple.q_des[6] = -0.3;    // 左后机身关节    右后机身关节
    joint_command_simple.q_des[7] = 1.2;     // 左后大腿关节    右后大腿关节
    joint_command_simple.q_des[8] = -2.721;  // 左后小腿关节    右后小腿关节
    joint_command_simple.q_des[9] = 0.3;     // 右后机身关节    左后机身关节
    joint_command_simple.q_des[10] = 1.2;    // 右后大腿关节    左后大腿关节
    joint_command_simple.q_des[11] = -2.721; // 右后小腿关节    左后小腿关节

    std::cout << "SET NOMINAL POSE" << std::endl;
}

// 此函数是发送底层控制指令的回调函数，DDS 会以一定频率触发该回调函数。此回调函数从上到下实现的功能分别是：①在线程第一次执行时，将当前各关节角度设置为目标角度，并初始化 Y、A、B、L2 四个按键的键值；②一大段安全冗余代码，确保 Go2 机器人在安全姿态下执行神经网络模型的输出；③计算 CRC 校验码，并调用 lowcmd_publisher的Write()函数将控制命令发送给 Go2 机器人。
void Custom::LowCmdWrite() // 阅读完成 √
{
    motiontime++; // 底层控制指令的下发次数

    if (firstRun && leg_control_lcm_data.q[0] != 0)
    {
        for (int i = 0; i < 12; i++)
        {
            // 程序首次运行至此的时候
            // 将当前各关节角度设置为目标角度
            joint_command_simple.q_des[i] = leg_control_lcm_data.q[i];
            // 初始化L2+B，防止damping被误触发
            key.components.Y = 0;
            key.components.A = 0;
            key.components.B = 0;
            key.components.L2 = 0;
        }
        firstRun = false;
    }

    // 写了一段安全冗余代码
    // 当roll角超过限制，或pitch角超过限制，或遥控器按下L2+B键
    // if (  low_state.imu_state().rpy()[0] > 0.5 || low_state.imu_state().rpy()[1] > 0.5 || ((int)key.components.B==1 && (int)key.components.L2==1))
    if (std::abs(low_state.imu_state().rpy()[0]) > 0.8 || std::abs(low_state.imu_state().rpy()[1]) > 0.8 || ((int)key.components.B == 1 && (int)key.components.L2 == 1))
    {
        for (int i = 0; i < 12; i++)
        {
            // 进入damping模式
            low_cmd.motor_cmd()[i].q() = 0;
            low_cmd.motor_cmd()[i].dq() = 0;
            low_cmd.motor_cmd()[i].kp() = 0;
            low_cmd.motor_cmd()[i].kd() = 5;
            low_cmd.motor_cmd()[i].tau() = 0;
        }
        std::cout << "======= Switched to Damping Mode, and the thread is sleeping ========" << std::endl;
        sleep(1.5);

        while (true)
        {

            // sleep(0.1);

            if (((int)key.components.B == 1 && (int)key.components.L2 == 1))
            {
                // [L2+B] is pressed again
                std::cout << "======= [L2+B] is pressed again, the script is about to exit========" << std::endl;
                exit(0);
            }
            else if (((int)key.components.A == 1 && (int)key.components.L2 == 1))
            {
                rsc.ServiceSwitch("sport_mode", 1);
                std::cout << "======= activate sport_mode service and exit========" << std::endl;
                sleep(0.5);
                exit(0);
            }
            else
            {
                if (((int)key.components.Y == 1 && (int)key.components.L2 == 1))
                {
                    std::cout << "=======  Switch to Walk These Ways ========" << std::endl;
                    std::cout << "Communicatino is set up successfully" << std::endl;
                    std::cout << "LCM <<<------------>>> Unitree SDK2" << std::endl;
                    std::cout << "------------------------------------" << std::endl;
                    std::cout << "------------------------------------" << std::endl;
                    std::cout << "Press L2+B if any unexpected error occurs" << std::endl;
                    break;
                }
                else
                {
                    std::cout << "======= Press [L2+B] again to exit ========" << std::endl;
                    std::cout << "======= Press [L2+Y] again to switch to WTW ========" << std::endl;
                    std::cout << "======= Press [L2+A] again to activate sport_mode service========" << std::endl;
                    sleep(0.01);
                }
            }
        }
    }
    else
    {
        for (int i = 0; i < 12; i++)
        {
            // 在确保安全的前提下，才执行神经网络模型的输出
            low_cmd.motor_cmd()[i].q() = joint_command_simple.q_des[i];
            low_cmd.motor_cmd()[i].dq() = joint_command_simple.qd_des[i];
            low_cmd.motor_cmd()[i].kp() = joint_command_simple.kp[i];
            low_cmd.motor_cmd()[i].kd() = joint_command_simple.kd[i];
            low_cmd.motor_cmd()[i].tau() = joint_command_simple.tau_ff[i];
        }
    }

    /*此段代码中第一行首先计算了 CRC 校验码。
    最后一行代码表示调用 lowcmd_publisher的Write()函数将控制命令发送给 Go2 机器人。*/
    low_cmd.crc() = crc32_core((uint32_t *)&low_cmd, (sizeof(unitree_go::msg::dds_::LowCmd_) >> 2) - 1);
    lowcmd_publisher->Write(low_cmd); // **向 Go2 机器人发送底层控制指令**
}

//
// 与循环工作的线程相关的函数定义已完结
//----------------------------------------------------------------------

void Custom::Init() // 阅读完成 √
{
    firstRun = true;
    InitLowCmd();
    SetNominalPose();

    // 这里决定了调用lc.handle()的时候，订阅什么消息，进行什么操作
    // 订阅什么消息："pd_plustau_targets"
    // 进行什么操作： lcm_receive_Handler
    lc.subscribe("pd_plustau_targets", &Custom::lcm_receive_Handler, this);

    /*create low_cmd publisher*/
    lowcmd_publisher.reset(new unitree::robot::ChannelPublisher<unitree_go::msg::dds_::LowCmd_>(TOPIC_LOWCMD));
    lowcmd_publisher->InitChannel();

    /*create low_state dds subscriber*/
    lowstate_subscriber.reset(new unitree::robot::ChannelSubscriber<unitree_go::msg::dds_::LowState_>(TOPIC_LOWSTATE));
    lowstate_subscriber->InitChannel(std::bind(&Custom::LowStateMessageHandler, this, std::placeholders::_1), 1);

    /*create joystick dds subscriber*/
    joystick_suber.reset(new unitree::robot::ChannelSubscriber<unitree_go::msg::dds_::WirelessController_>(TOPIC_JOYSTICK));
    joystick_suber->InitChannel(std::bind(&Custom::JoystickHandler, this, std::placeholders::_1), 1);
}

void Custom::Loop() // 阅读完成 √
{
    // 新增线程可以实现loop function的功能

    // intervalMicrosec : 1微秒 = 0.000001秒
    // 当dt=0.002s
    // ntervalMicrosec = 2000us

    /*lcm send thread*/
    LcmSendThreadPtr = unitree::common::CreateRecurrentThreadEx("lcm_send_thread", UT_CPU_ID_NONE, dt * 1e6, &Custom::lcm_send, this);
    /*lcm receive thread*/
    LcmRecevThreadPtr = unitree::common::CreateRecurrentThreadEx("lcm_recev_thread", UT_CPU_ID_NONE, dt * 1e6, &Custom::lcm_receive, this);
    /*low command write thread*/
    lowCmdWriteThreadPtr = unitree::common::CreateRecurrentThreadEx("dds_write_thread", UT_CPU_ID_NONE, dt * 1e6, &Custom::LowCmdWrite, this);
}

int main(int argc, char **argv) // 阅读完成 √
{
    if (argc < 2)
    {
        std::cout << "Usage: ./" << argv[0] << " networkInterface" << std::endl;
        exit(-1);
    }

    std::cout << "Control level is set to LOW-level." << std::endl
              << "WARNING: Make sure the robot is lie down." << std::endl
              << "Caution: “sport_mode” service will be shut down." << std::endl
              << "Press Enter to continue." << std::endl;
    std::cin.ignore();

    unitree::robot::ChannelFactory::Instance()->Init(0, argv[1]); // 传入本机的网卡地址（PC or Jetson Orin NX）

    Custom custom;
    custom.InitRobotStateClient();

    // 关闭 Go2 的主运控服务（sport_mode）
    if (custom.QueryServiceStatus("sport_mode"))
    {
        std::cout << "Try to deactivate the service: " << "sport_mode" << std::endl;
        custom.SwitchService("sport_mode", 0);
        sleep(1);
        if (!custom.QueryServiceStatus("sport_mode"))
        {
            std::cout << "Service: " << "sport_mode" << " has been deactivated successfully." << std::endl;
            std::cout << "Next step is to set up LCM communication." << std::endl
                      << "Press Enter to continue." << std::endl;
            std::cin.ignore();
        }
    }
    else
    {
        std::cout << "\"sport_mode\" service is already inactive now." << std::endl
                  << "Next step is to set up LCM communication." << std::endl
                  << "Press Enter to continue." << std::endl;
        std::cin.ignore();
    }

    custom.Init();

    // 未做修改
    std::cout << "Communication is set up successfully" << std::endl;
    std::cout << "LCM <<<------------>>> Unitree SDK2" << std::endl;
    std::cout << "------------------------------------" << std::endl;
    std::cout << "------------------------------------" << std::endl;
    std::cout << "Press L2+B if any unexpected error occurs" << std::endl;
    // 未做修改

    custom.Loop();

    while (true)
    {
        sleep(10);
    }

    return 0;
}

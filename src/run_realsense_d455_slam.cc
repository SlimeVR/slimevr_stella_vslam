#ifdef HAVE_PANGOLIN_VIEWER
#include "pangolin_viewer/viewer.h"
#endif
#ifdef HAVE_IRIDESCENCE_VIEWER
#include "iridescence_viewer/viewer.h"
#endif
#ifdef HAVE_SOCKET_PUBLISHER
#include "socket_publisher/publisher.h"
#endif

#include "stella_vslam/system.h"
#include "stella_vslam/config.h"
#include "stella_vslam/camera/base.h"
#include "stella_vslam/util/yaml.h"

#include <iostream>
#include <chrono>
#include <numeric>
#include <fstream>
#include <thread>

#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <spdlog/spdlog.h>
#include <popl.hpp>

#include <librealsense2/rs.hpp>

#include <ghc/filesystem.hpp>
namespace fs = ghc::filesystem;

#ifdef USE_STACK_TRACE_LOGGER
#include <backward.hpp>
#endif

#ifdef USE_GOOGLE_PERFTOOLS
#include <gperftools/profiler.h>
#endif

// Generate a temporary YAML config file with RealSense intrinsics
std::string generate_realsense_config(const rs2::pipeline_profile& profile, 
                                     const std::string& camera_name,
                                     const std::string& setup_type = "RGBD") {
    auto color_stream = profile.get_stream(RS2_STREAM_COLOR).as<rs2::video_stream_profile>();
    auto depth_stream = profile.get_stream(RS2_STREAM_DEPTH).as<rs2::video_stream_profile>();
    
    auto color_intrinsics = color_stream.get_intrinsics();
    auto depth_intrinsics = depth_stream.get_intrinsics();
    
    // Get extrinsics between color and depth
    auto depth_to_color = depth_stream.get_extrinsics_to(color_stream);
    
    // Create temporary config file
    std::string temp_config_path = "/tmp/realsense_d455_config.yaml";
    std::ofstream config_file(temp_config_path);
    
    config_file << "# Auto-generated Intel RealSense D455 Configuration\n\n";
    config_file << "#==============#\n";
    config_file << "# Camera Model #\n";
    config_file << "#==============#\n\n";
    
    config_file << "Camera:\n";
    config_file << "  name: \"" << camera_name << "\"\n";
    config_file << "  setup: \"" << setup_type << "\"\n";
    config_file << "  model: \"perspective\"\n\n";
    
    config_file << "  fps: 30.0\n";
    config_file << "  cols: " << color_intrinsics.width << "\n";
    config_file << "  rows: " << color_intrinsics.height << "\n";
    config_file << "  color_order: \"RGB\"\n\n";
    
    // Intrinsic parameters
    config_file << "  # Intrinsic parameters\n";
    config_file << "  fx: " << color_intrinsics.fx << "\n";
    config_file << "  fy: " << color_intrinsics.fy << "\n";
    config_file << "  cx: " << color_intrinsics.ppx << "\n";
    config_file << "  cy: " << color_intrinsics.ppy << "\n\n";
    
    // Distortion parameters
    config_file << "  # Distortion parameters\n";
    config_file << "  k1: " << color_intrinsics.coeffs[0] << "\n";
    config_file << "  k2: " << color_intrinsics.coeffs[1] << "\n";
    config_file << "  p1: " << color_intrinsics.coeffs[2] << "\n";
    config_file << "  p2: " << color_intrinsics.coeffs[3] << "\n";
    config_file << "  k3: " << color_intrinsics.coeffs[4] << "\n\n";
    
    if (setup_type == "RGBD") {
        // RGBD-specific parameters
        config_file << "  # RGBD-specific parameters\n";
        config_file << "  focal_x_baseline: " << depth_intrinsics.fx << "\n";
        config_file << "  depth_threshold: 10.0\n\n";
    }
    
    config_file << "#==================#\n";
    config_file << "# Preprocessing    #\n";
    config_file << "#==================#\n\n";
    
    config_file << "Preprocessing:\n";
    config_file << "  min_size: 800\n";
    if (setup_type == "RGBD") {
        config_file << "  depthmap_factor: 0.001\n\n";  // RealSense outputs in mm, convert to m
    } else {
        config_file << "\n";
    }
    
    config_file << "#================#\n";
    config_file << "# ORB Parameters #\n";
    config_file << "#================#\n\n";
    
    config_file << "Feature:\n";
    config_file << "  name: \"RealSense D455 ORB feature extraction\"\n";
    config_file << "  scale_factor: 1.2\n";
    config_file << "  num_levels: 8\n";
    config_file << "  ini_fast_threshold: 20\n";
    config_file << "  min_fast_threshold: 7\n\n";
    
    config_file << "#===================#\n";
    config_file << "# Mapping Parameters#\n";
    config_file << "#===================#\n\n";
    
    config_file << "Mapping:\n";
    if (setup_type == "RGBD") {
        config_file << "  baseline_dist_thr: 0.07471049682\n";  // Based on typical RGBD configs
    } else {
        config_file << "  baseline_dist_thr_ratio: 0.02\n";
    }
    config_file << "  redundant_obs_ratio_thr: 0.9\n\n";
    
    // Add optional viewer configurations
    config_file << "#===================#\n";
    config_file << "# Viewer Parameters #\n";
    config_file << "#===================#\n\n";
    
    config_file << "PangolinViewer:\n";
    config_file << "  keyframe_size: 0.05\n";
    config_file << "  keyframe_line_width: 1\n";
    config_file << "  graph_line_width: 1\n";
    config_file << "  point_size: 2\n";
    config_file << "  camera_size: 0.08\n";
    config_file << "  camera_line_width: 3\n";
    config_file << "  viewpoint_x: 0.0\n";
    config_file << "  viewpoint_y: -0.9\n";
    config_file << "  viewpoint_z: -1.9\n";
    config_file << "  viewpoint_f: 400.0\n\n";
    
    config_file.close();
    
    spdlog::info("Generated RealSense config file: {}", temp_config_path);
    spdlog::info("Color intrinsics: fx={}, fy={}, cx={}, cy={}", 
                 color_intrinsics.fx, color_intrinsics.fy, 
                 color_intrinsics.ppx, color_intrinsics.ppy);
    spdlog::info("Image resolution: {}x{}", color_intrinsics.width, color_intrinsics.height);
    
    return temp_config_path;
}

int rgbd_tracking(const std::shared_ptr<stella_vslam::system>& slam,
                  const std::shared_ptr<stella_vslam::config>& cfg,
                  const std::string& mask_img_path,
                  const float scale,
                  const std::string& map_db_path,
                  const std::string& viewer_string,
                  const int width = 1280,
                  const int height = 720,
                  const int fps = 30) {
    
    // Load the mask image
    const cv::Mat mask = mask_img_path.empty() ? cv::Mat{} : cv::imread(mask_img_path, cv::IMREAD_GRAYSCALE);

    // Initialize RealSense pipeline
    rs2::pipeline pipe;
    rs2::config rs_config;
    
    // Configure RealSense streams
    rs_config.enable_stream(RS2_STREAM_COLOR, width, height, RS2_FORMAT_BGR8, fps);
    rs_config.enable_stream(RS2_STREAM_DEPTH, width, height, RS2_FORMAT_Z16, fps);
    
    // Start streaming
    auto pipeline_profile = pipe.start(rs_config);
    
    // Create viewer objects
#ifdef HAVE_PANGOLIN_VIEWER
    std::shared_ptr<pangolin_viewer::viewer> viewer;
    if (viewer_string == "pangolin_viewer") {
        viewer = std::make_shared<pangolin_viewer::viewer>(
            stella_vslam::util::yaml_optional_ref(cfg->yaml_node_, "PangolinViewer"),
            slam,
            slam->get_frame_publisher(),
            slam->get_map_publisher());
    }
#endif
#ifdef HAVE_IRIDESCENCE_VIEWER
    std::shared_ptr<iridescence_viewer::viewer> iridescence_viewer;
    std::mutex mtx_pause;
    bool is_paused = false;
    std::mutex mtx_terminate;
    bool terminate_is_requested = false;
    std::mutex mtx_step;
    unsigned int step_count = 0;
    if (viewer_string == "iridescence_viewer") {
        iridescence_viewer = std::make_shared<iridescence_viewer::viewer>(
            stella_vslam::util::yaml_optional_ref(cfg->yaml_node_, "IridescenceViewer"),
            slam->get_frame_publisher(),
            slam->get_map_publisher());
        iridescence_viewer->add_checkbox("Pause", [&is_paused, &mtx_pause](bool check) {
            std::lock_guard<std::mutex> lock(mtx_pause);
            is_paused = check;
        });
        iridescence_viewer->add_button("Step", [&step_count, &mtx_step] {
            std::lock_guard<std::mutex> lock(mtx_step);
            step_count++;
        });
        iridescence_viewer->add_button("Reset", [&slam] {
            slam->request_reset();
        });
        iridescence_viewer->add_button("Save and exit", [&is_paused, &mtx_pause, &terminate_is_requested, &mtx_terminate, &slam, &iridescence_viewer] {
            std::lock_guard<std::mutex> lock1(mtx_pause);
            is_paused = false;
            std::lock_guard<std::mutex> lock2(mtx_terminate);
            terminate_is_requested = true;
            iridescence_viewer->request_terminate();
        });
        iridescence_viewer->add_close_callback([&is_paused, &mtx_pause, &terminate_is_requested, &mtx_terminate] {
            std::lock_guard<std::mutex> lock1(mtx_pause);
            is_paused = false;
            std::lock_guard<std::mutex> lock2(mtx_terminate);
            terminate_is_requested = true;
        });
    }
#endif
#ifdef HAVE_SOCKET_PUBLISHER
    std::shared_ptr<socket_publisher::publisher> publisher;
    if (viewer_string == "socket_publisher") {
        publisher = std::make_shared<socket_publisher::publisher>(
            stella_vslam::util::yaml_optional_ref(cfg->yaml_node_, "SocketPublisher"),
            slam,
            slam->get_frame_publisher(),
            slam->get_map_publisher());
    }
#endif

    // Create alignment object for aligning depth to color
    rs2::align align_to_color(RS2_STREAM_COLOR);
    
    cv::Mat color_frame, depth_frame;
    std::vector<double> track_times;
    unsigned int num_frame = 0;

    bool is_not_end = true;
    // Run SLAM in another thread
    std::thread thread([&]() {
        while (is_not_end) {
#ifdef HAVE_IRIDESCENCE_VIEWER
            while (true) {
                {
                    std::lock_guard<std::mutex> lock(mtx_pause);
                    if (!is_paused) {
                        break;
                    }
                }
                {
                    std::lock_guard<std::mutex> lock(mtx_step);
                    if (step_count > 0) {
                        step_count--;
                        break;
                    }
                }
                std::this_thread::sleep_for(std::chrono::microseconds(5000));
            }
#endif

#ifdef HAVE_IRIDESCENCE_VIEWER
            // Check if termination is requested
            {
                std::lock_guard<std::mutex> lock(mtx_terminate);
                if (terminate_is_requested) {
                    break;
                }
            }
#else
            if (slam->terminate_is_requested()) {
                break;
            }
#endif

            try {
                // Wait for frames
                rs2::frameset frames = pipe.wait_for_frames();
                
                // Align depth to color
                auto aligned_frames = align_to_color.process(frames);
                
                auto color = aligned_frames.get_color_frame();
                auto depth = aligned_frames.get_depth_frame();
                
                if (!color || !depth) {
                    continue;
                }

                // Convert to OpenCV format
                color_frame = cv::Mat(cv::Size(color.get_width(), color.get_height()), 
                                    CV_8UC3, (void*)color.get_data(), cv::Mat::AUTO_STEP);
                depth_frame = cv::Mat(cv::Size(depth.get_width(), depth.get_height()), 
                                    CV_16UC1, (void*)depth.get_data(), cv::Mat::AUTO_STEP);

                // Convert BGR to RGB
                cv::cvtColor(color_frame, color_frame, cv::COLOR_BGR2RGB);

                if (color_frame.empty() || depth_frame.empty()) {
                    continue;
                }

                // Apply scaling if needed
                if (scale != 1.0) {
                    cv::resize(color_frame, color_frame, cv::Size(), scale, scale, cv::INTER_LINEAR);
                    cv::resize(depth_frame, depth_frame, cv::Size(), scale, scale, cv::INTER_NEAREST);
                }

                const auto tp_1 = std::chrono::steady_clock::now();

                // Get timestamp
                double timestamp = color.get_timestamp() * 0.001; // Convert from ms to seconds

                // Feed RGBD frame to SLAM system
                slam->feed_RGBD_frame(color_frame, depth_frame, timestamp, mask);

                const auto tp_2 = std::chrono::steady_clock::now();

                const auto track_time = std::chrono::duration_cast<std::chrono::duration<double>>(tp_2 - tp_1).count();
                track_times.push_back(track_time);

                ++num_frame;
            }
            catch (const rs2::error& e) {
                spdlog::error("RealSense error: {}", e.what());
                is_not_end = false;
                break;
            }
            catch (const std::exception& e) {
                spdlog::error("Error: {}", e.what());
                is_not_end = false;
                break;
            }
        }

        // Wait until loop BA is finished
        while (slam->loop_BA_is_running()) {
            std::this_thread::sleep_for(std::chrono::microseconds(5000));
        }
    });

    // Run viewer in the current thread
    if (viewer_string == "pangolin_viewer") {
#ifdef HAVE_PANGOLIN_VIEWER
        viewer->run();
#endif
    }
    if (viewer_string == "iridescence_viewer") {
#ifdef HAVE_IRIDESCENCE_VIEWER
        iridescence_viewer->run();
#endif
    }
    if (viewer_string == "socket_publisher") {
#ifdef HAVE_SOCKET_PUBLISHER
        publisher->run();
#endif
    }

    thread.join();

    // Stop RealSense pipeline
    pipe.stop();

    // Shutdown SLAM system
    slam->shutdown();

    // Print timing statistics
    if (!track_times.empty()) {
        std::sort(track_times.begin(), track_times.end());
        const auto total_track_time = std::accumulate(track_times.begin(), track_times.end(), 0.0);
        std::cout << "median tracking time: " << track_times.at(track_times.size() / 2) << "[s]" << std::endl;
        std::cout << "mean tracking time: " << total_track_time / track_times.size() << "[s]" << std::endl;
    }

    // Save map if requested
    if (!map_db_path.empty()) {
        if (!slam->save_map_database(map_db_path)) {
            return EXIT_FAILURE;
        }
    }

    return EXIT_SUCCESS;
}

int main(int argc, char* argv[]) {
#ifdef USE_STACK_TRACE_LOGGER
    backward::SignalHandling sh;
#endif

    // Create options
    popl::OptionParser op("Allowed options");
    auto help = op.add<popl::Switch>("h", "help", "produce help message");
    auto vocab_file_path = op.add<popl::Value<std::string>>("v", "vocab", "vocabulary file path");
    auto without_vocab = op.add<popl::Switch>("", "without-vocab", "run without vocabulary file");
    auto config_file_path = op.add<popl::Value<std::string>>("c", "config", "config file path (optional - will auto-generate if not provided)", "");
    auto mask_img_path = op.add<popl::Value<std::string>>("", "mask", "mask image path", "");
    auto scale = op.add<popl::Value<float>>("s", "scale", "scaling ratio of images", 1.0);
    auto map_db_path_in = op.add<popl::Value<std::string>>("i", "map-db-in", "load a map from this path", "");
    auto map_db_path_out = op.add<popl::Value<std::string>>("o", "map-db-out", "store a map database at this path after slam", "");
    auto log_level = op.add<popl::Value<std::string>>("", "log-level", "log level", "info");
    auto disable_mapping = op.add<popl::Switch>("", "disable-mapping", "disable mapping");
    auto temporal_mapping = op.add<popl::Switch>("", "temporal-mapping", "enable temporal mapping");
    auto viewer = op.add<popl::Value<std::string>>("", "viewer", "viewer [iridescence_viewer, pangolin_viewer, socket_publisher, none]");
    auto width = op.add<popl::Value<int>>("w", "width", "image width", 1280);
    auto height = op.add<popl::Value<int>>("", "height", "image height", 720);
    auto fps = op.add<popl::Value<int>>("f", "fps", "frames per second", 30);
    
    try {
        op.parse(argc, argv);
    }
    catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        std::cerr << std::endl;
        std::cerr << op << std::endl;
        return EXIT_FAILURE;
    }

    // Check validness of options
    if (help->is_set()) {
        std::cerr << op << std::endl;
        return EXIT_FAILURE;
    }
    if (!op.unknown_options().empty()) {
        for (const auto& unknown_option : op.unknown_options()) {
            std::cerr << "unknown_options: " << unknown_option << std::endl;
        }
        std::cerr << op << std::endl;
        return EXIT_FAILURE;
    }
    if ((!vocab_file_path->is_set() && !without_vocab->is_set())) {
        std::cerr << "Either --vocab or --without-vocab must be specified" << std::endl;
        std::cerr << std::endl;
        std::cerr << op << std::endl;
        return EXIT_FAILURE;
    }

    // Viewer selection
    std::string viewer_string;
    if (viewer->is_set()) {
        viewer_string = viewer->value();
        if (viewer_string != "pangolin_viewer"
            && viewer_string != "socket_publisher"
            && viewer_string != "iridescence_viewer"
            && viewer_string != "none") {
            std::cerr << "invalid arguments (--viewer)" << std::endl
                      << std::endl
                      << op << std::endl;
            return EXIT_FAILURE;
        }
#ifndef HAVE_PANGOLIN_VIEWER
        if (viewer_string == "pangolin_viewer") {
            std::cerr << "pangolin_viewer not linked" << std::endl
                      << std::endl
                      << op << std::endl;
            return EXIT_FAILURE;
        }
#endif
#ifndef HAVE_IRIDESCENCE_VIEWER
        if (viewer_string == "iridescence_viewer") {
            std::cerr << "iridescence_viewer not linked" << std::endl
                      << std::endl
                      << op << std::endl;
            return EXIT_FAILURE;
        }
#endif
#ifndef HAVE_SOCKET_PUBLISHER
        if (viewer_string == "socket_publisher") {
            std::cerr << "socket_publisher not linked" << std::endl
                      << std::endl
                      << op << std::endl;
            return EXIT_FAILURE;
        }
#endif
    }
    else {
#ifdef HAVE_IRIDESCENCE_VIEWER
        viewer_string = "iridescence_viewer";
#elif defined(HAVE_PANGOLIN_VIEWER)
        viewer_string = "pangolin_viewer";
#elif defined(HAVE_SOCKET_PUBLISHER)
        viewer_string = "socket_publisher";
#endif
    }

    // Setup logger
    spdlog::set_pattern("[%Y-%m-%d %H:%M:%S.%e] %^[%L] %v%$");
    spdlog::set_level(spdlog::level::from_str(log_level->value()));

    // Initialize RealSense to get intrinsics for config generation
    std::string actual_config_file_path;
    if (config_file_path->is_set() && !config_file_path->value().empty()) {
        actual_config_file_path = config_file_path->value();
        spdlog::info("Using provided config file: {}", actual_config_file_path);
    } else {
        spdlog::info("Auto-generating RealSense D455 configuration...");
        
        try {
            // Initialize RealSense pipeline temporarily to get intrinsics
            rs2::pipeline temp_pipe;
            rs2::config temp_config;
            temp_config.enable_stream(RS2_STREAM_COLOR, width->value(), height->value(), RS2_FORMAT_BGR8, fps->value());
            temp_config.enable_stream(RS2_STREAM_DEPTH, width->value(), height->value(), RS2_FORMAT_Z16, fps->value());
            
            auto temp_profile = temp_pipe.start(temp_config);
            actual_config_file_path = generate_realsense_config(temp_profile, "Intel RealSense D455", "RGBD");
            temp_pipe.stop();
        }
        catch (const rs2::error& e) {
            spdlog::critical("Failed to initialize RealSense D455: {}", e.what());
            spdlog::critical("Make sure the RealSense D455 is connected and accessible");
            return EXIT_FAILURE;
        }
    }

    // Load configuration
    std::shared_ptr<stella_vslam::config> cfg;
    try {
        cfg = std::make_shared<stella_vslam::config>(actual_config_file_path);
    }
    catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }

#ifdef USE_GOOGLE_PERFTOOLS
    ProfilerStart("slam.prof");
#endif

    // Build SLAM system
    std::string vocab_file_path_str = (without_vocab->is_set()) ? "" : vocab_file_path->value();
    auto slam = std::make_shared<stella_vslam::system>(cfg, vocab_file_path_str);
    bool need_initialize = true;
    if (map_db_path_in->is_set()) {
        need_initialize = false;
        const auto path = fs::path(map_db_path_in->value());
        if (path.extension() == ".yaml") {
            YAML::Node node = YAML::LoadFile(path);
            for (const auto& map_path : node["maps"].as<std::vector<std::string>>()) {
                if (!slam->load_map_database(path.parent_path() / map_path)) {
                    return EXIT_FAILURE;
                }
            }
        }
        else {
            if (!slam->load_map_database(path)) {
                return EXIT_FAILURE;
            }
        }
    }
    slam->startup(need_initialize);
    if (disable_mapping->is_set()) {
        slam->disable_mapping_module();
    }
    else if (temporal_mapping->is_set()) {
        slam->enable_temporal_mapping();
        slam->disable_loop_detector();
    }

    // Run RGBD tracking
    int ret = rgbd_tracking(slam,
                           cfg,
                           mask_img_path->value(),
                           scale->value(),
                           map_db_path_out->value(),
                           viewer_string,
                           width->value(),
                           height->value(),
                           fps->value());

#ifdef USE_GOOGLE_PERFTOOLS
    ProfilerStop();
#endif

    return ret;
}
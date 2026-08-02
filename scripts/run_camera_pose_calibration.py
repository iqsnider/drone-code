from calibration.calibrate_aruco import main

if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\ninterrupted")
        raise SystemExit(130)

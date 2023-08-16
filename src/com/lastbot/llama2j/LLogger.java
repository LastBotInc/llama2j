package com.lastbot.llama2j;

import java.text.Format;
import java.text.SimpleDateFormat;
import java.util.Date;

public class LLogger {
    private static final Format formatter = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss");

    public static synchronized void internalError(String message) {
        printError("INTERNAL ERROR: ", message, null);
    }

    public static synchronized void internalError(String message, Throwable t) {
        printError("INTERNAL ERROR: ", message, t);
    }

    public static synchronized void error(String message) {
        printError("ERROR", message, null);
    }

    public static synchronized void error(String message, Throwable t) {
        printError("ERROR: ", message, t);
    }

    public static synchronized void warning(String message) {
        print("WARNING: " + timeStamp() + " " + message);
    }

    public static synchronized void info(String message) {
        print("INFO: " + timeStamp() + " " + message);
    }

    private static final String TIME_NAME_FORMAT = "%-40s";

    public static synchronized void time(String task, long start, long end) {
        String message = String.format(TIME_NAME_FORMAT, task) + " " + String.format("%,d", (end-start)) + " ms";
        print("TIME: " + timeStamp() + " " + message);
    }

    public static synchronized void debug(String message) {
        print("DEBUG: " + timeStamp() + " " + message);
    }

    private static String timeStamp() {
        return formatter.format(new Date());
    }

    private static void printError(String type, String message, Throwable t) {
        print(type + " " + timeStamp() + " " + message + (t == null ? "" : t.toString()));

        if (t != null) {
            for (StackTraceElement ste : t.getStackTrace()) {
                String s = ste.toString();
                if (!s.contains("LLogger") &&
                        !s.contains("java.lang.Thread.getStackTrace")) {
                    print("\tat " + ste);
                }

            }
        }
    }

    private static void print(String s) {
        System.out.println(s);
        System.out.flush();
    }
}

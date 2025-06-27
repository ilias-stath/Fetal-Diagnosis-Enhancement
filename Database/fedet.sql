-- phpMyAdmin SQL Dump
-- version 5.2.1
-- https://www.phpmyadmin.net/
--
-- Εξυπηρετητής: 127.0.0.1
-- Χρόνος δημιουργίας: 27 Ιουν 2025 στις 20:35:16
-- Έκδοση διακομιστή: 10.4.28-MariaDB
-- Έκδοση PHP: 8.2.4

SET SQL_MODE = "NO_AUTO_VALUE_ON_ZERO";
START TRANSACTION;
SET time_zone = "+00:00";


/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8mb4 */;

--
-- Βάση δεδομένων: `fedet`
--

-- --------------------------------------------------------

--
-- Δομή πίνακα για τον πίνακα `administrators`
--

CREATE TABLE `administrators` (
  `id` int(11) NOT NULL,
  `user_id` int(11) NOT NULL,
  `clearance` varchar(10) DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Άδειασμα δεδομένων του πίνακα `administrators`
--

INSERT INTO `administrators` (`id`, `user_id`, `clearance`) VALUES
(1, 1, 'A9'),
(3, 4, 'A9');

-- --------------------------------------------------------

--
-- Δομή πίνακα για τον πίνακα `medical_personnel`
--

CREATE TABLE `medical_personnel` (
  `id` int(11) NOT NULL,
  `user_id` int(11) NOT NULL,
  `specialization` varchar(50) DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Άδειασμα δεδομένων του πίνακα `medical_personnel`
--

INSERT INTO `medical_personnel` (`id`, `user_id`, `specialization`) VALUES
(1, 2, 'Chiroopractor'),
(2, 3, 'Chinese Tradinional Medicine');

-- --------------------------------------------------------

--
-- Δομή πίνακα για τον πίνακα `model`
--

CREATE TABLE `model` (
  `id` int(11) NOT NULL,
  `name` varchar(255) NOT NULL,
  `parameters` text NOT NULL,
  `valuesN` text NOT NULL,
  `maker` int(11) DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

-- --------------------------------------------------------

--
-- Δομή πίνακα για τον πίνακα `results`
--

CREATE TABLE `results` (
  `id` int(11) NOT NULL,
  `Patient_Name` varchar(255) NOT NULL,
  `Fetal_Health` enum('Normal','Suspect','Pathological') NOT NULL,
  `medical_supervisor` int(11) DEFAULT NULL,
  `parameters` longtext CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL CHECK (json_valid(`parameters`)),
  `date` timestamp NOT NULL DEFAULT current_timestamp() ON UPDATE current_timestamp(),
  `model_id` int(11) DEFAULT NULL,
  `image1` mediumblob DEFAULT NULL,
  `image2` mediumblob DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Άδειασμα δεδομένων του πίνακα `results`
--

INSERT INTO `results` (`id`, `Patient_Name`, `Fetal_Health`, `medical_supervisor`, `parameters`, `date`, `model_id`, `image1`, `image2`) VALUES
(1, 'test subject 1', 'Normal', 1, '{\"parameter1\": 120, \"parameter2\": 75, \"parameter3\": 98}', '2025-06-14 20:38:05', NULL, NULL, NULL),
(2, 'test subject 2', 'Pathological', 2, '{\"parameter1\": 10, \"parameter2\": 0, \"parameter3\": 9}', '2025-06-14 20:40:25', NULL, NULL, NULL),
(3, 'test subject 2', 'Normal', 2, '{\"parameter1\": 100, \"parameter2\": 100, \"parameter3\": 90}', '2025-06-14 20:45:42', NULL, NULL, NULL);

-- --------------------------------------------------------

--
-- Δομή πίνακα για τον πίνακα `users`
--

CREATE TABLE `users` (
  `id` int(11) NOT NULL,
  `fullName` varchar(50) NOT NULL,
  `username` varchar(15) NOT NULL,
  `password` varchar(255) NOT NULL,
  `role` enum('medical','admin') NOT NULL,
  `telephone` varchar(20) NOT NULL,
  `email` varchar(255) NOT NULL,
  `address` varchar(255) NOT NULL,
  `created_at` timestamp NOT NULL DEFAULT current_timestamp()
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Άδειασμα δεδομένων του πίνακα `users`
--

INSERT INTO `users` (`id`, `fullName`, `username`, `password`, `role`, `telephone`, `email`, `address`, `created_at`) VALUES
(1, 'Ilias Stathakos', 'ilias_stath', '$2b$12$ttAa9ABZYuO7PhmyVPPt/evLRbZAgH77vklODsNHnJgohmk4K5dE.', 'admin', '+306999999', 'ece002017@uowm.gr', 'kozani', '2025-06-26 20:15:42'),
(2, 'Konstantinos Papathanasiou', 'konnos_pap', '$2b$12$5YVvuG5PZqcosTw0W8pF5O56ES3Ey.tcUSDeuOJSQRTWTUR6sBfvO', 'medical', '+3068888888', 'ece02008@uowm.gr', 'Kozani', '2025-06-14 20:04:19'),
(3, 'Georgios Ktistakis', 'george_ktist', '$2b$12$y/iujqomXl7UteML0T1Qu.tPBFL6RLnP7286sX8u0I5b5wcni66qS', 'medical', '+3067777777', 'ece01981@uowm.gr', 'kozani', '2025-06-14 20:06:12'),
(4, 'agasdgd', 'gasdqwef', '$2b$12$LCMCXfAzBXaoMoC3NIq82uy4U27J2nB32EDixofjFQxBc8MkMWU6C', 'admin', '+306916644999', 'ktist@', 'koza13ni', '2025-06-27 16:48:22');

--
-- Ευρετήρια για άχρηστους πίνακες
--

--
-- Ευρετήρια για πίνακα `administrators`
--
ALTER TABLE `administrators`
  ADD PRIMARY KEY (`id`),
  ADD UNIQUE KEY `user_id` (`user_id`);

--
-- Ευρετήρια για πίνακα `medical_personnel`
--
ALTER TABLE `medical_personnel`
  ADD PRIMARY KEY (`id`),
  ADD UNIQUE KEY `user_id` (`user_id`);

--
-- Ευρετήρια για πίνακα `model`
--
ALTER TABLE `model`
  ADD PRIMARY KEY (`id`),
  ADD KEY `supervisor` (`maker`);

--
-- Ευρετήρια για πίνακα `results`
--
ALTER TABLE `results`
  ADD PRIMARY KEY (`id`),
  ADD KEY `fk_medical_supervisor` (`medical_supervisor`),
  ADD KEY `fk_model_id` (`model_id`);

--
-- Ευρετήρια για πίνακα `users`
--
ALTER TABLE `users`
  ADD PRIMARY KEY (`id`),
  ADD UNIQUE KEY `username` (`username`);

--
-- AUTO_INCREMENT για άχρηστους πίνακες
--

--
-- AUTO_INCREMENT για πίνακα `administrators`
--
ALTER TABLE `administrators`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=4;

--
-- AUTO_INCREMENT για πίνακα `medical_personnel`
--
ALTER TABLE `medical_personnel`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=4;

--
-- AUTO_INCREMENT για πίνακα `model`
--
ALTER TABLE `model`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT;

--
-- AUTO_INCREMENT για πίνακα `results`
--
ALTER TABLE `results`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=4;

--
-- AUTO_INCREMENT για πίνακα `users`
--
ALTER TABLE `users`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=5;

--
-- Περιορισμοί για άχρηστους πίνακες
--

--
-- Περιορισμοί για πίνακα `administrators`
--
ALTER TABLE `administrators`
  ADD CONSTRAINT `administrators_ibfk_1` FOREIGN KEY (`user_id`) REFERENCES `users` (`id`) ON DELETE CASCADE;

--
-- Περιορισμοί για πίνακα `medical_personnel`
--
ALTER TABLE `medical_personnel`
  ADD CONSTRAINT `medical_personnel_ibfk_1` FOREIGN KEY (`user_id`) REFERENCES `users` (`id`) ON DELETE CASCADE;

--
-- Περιορισμοί για πίνακα `model`
--
ALTER TABLE `model`
  ADD CONSTRAINT `model_ibfk_1` FOREIGN KEY (`maker`) REFERENCES `medical_personnel` (`id`) ON DELETE SET NULL ON UPDATE CASCADE;

--
-- Περιορισμοί για πίνακα `results`
--
ALTER TABLE `results`
  ADD CONSTRAINT `fk_medical_supervisor` FOREIGN KEY (`medical_supervisor`) REFERENCES `medical_personnel` (`id`) ON DELETE SET NULL,
  ADD CONSTRAINT `fk_model_id` FOREIGN KEY (`model_id`) REFERENCES `model` (`id`) ON DELETE SET NULL ON UPDATE CASCADE;
COMMIT;

/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
